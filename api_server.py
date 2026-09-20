"""Serve the Ingrid HTML UI and a local Pinecone-backed product API."""

import base64
import binascii
import json
import re
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "ingredient-checker"))

from barcode_detector import extract_text
from chain import analyse_label, answer_question, get_openai_client


def decode_barcode_with_vision(image):
    """Use the configured vision-capable model only after conventional decoding fails."""
    client = get_openai_client()
    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read the barcode digits in this product image. Return only the 8 to 14 digit barcode, with no other text. Return NONE if no barcode digits are visible.",
                        },
                        {"type": "image_url", "image_url": {"url": image, "detail": "high"}},
                    ],
                }
            ],
            temperature=0,
        )
    except Exception:
        return ""

    match = re.fullmatch(r"\s*(\d{8,14})\s*", response.choices[0].message.content or "")
    candidate = match.group(1) if match else ""
    return candidate if len(candidate) >= 12 else ""


def decode_barcode_with_ocr(image_bytes):
    """Read a full-length printed barcode number when image decoders miss the bars."""
    try:
        import cv2
        import easyocr
        import numpy as np

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return ""
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variants = [
            image,
            cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
            cv2.resize(gray[int(height * 0.35):], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            cv2.resize(
                cv2.threshold(gray[int(height * 0.35):], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_NEAREST,
            ),
        ]
    except Exception:
        return ""

    candidates = []
    for variant in variants:
        try:
            results = reader.readtext(variant, detail=1, paragraph=False, allowlist="0123456789")
        except Exception:
            continue
        digit_boxes = []
        for box, text, confidence in results:
            digits = re.sub(r"\D", "", text)
            if digits and confidence >= 0.2:
                center_x = sum(point[0] for point in box) / len(box)
                center_y = sum(point[1] for point in box) / len(box)
                digit_boxes.append((center_x, center_y, digits))
                if 12 <= len(digits) <= 14:
                    candidates.append(digits)
        for _, _, line_digits in _group_ocr_digits(digit_boxes):
            if 12 <= len(line_digits) <= 14:
                candidates.append(line_digits)
    return max(candidates, key=len, default="")


def _group_ocr_digits(digit_boxes):
    """Join OCR digit fragments that share a horizontal barcode-number row."""
    groups = []
    for center_x, center_y, digits in sorted(digit_boxes, key=lambda item: item[1]):
        group = next((item for item in groups if abs(item[0] - center_y) < 80), None)
        if group is None:
            group = [center_y, []]
            groups.append(group)
        group[1].append((center_x, digits))
    return [
        (center_y, min(fragment[0] for fragment in fragments), "".join(digits for _, digits in sorted(fragments)))
        for center_y, fragments in groups
    ]


class IngridHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if urlparse(self.path).path == "/ingrid_your_ingredient_decoded.html":
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", "/")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/product", "/api/decode-barcode", "/api/chat"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON request")
            return

        if endpoint == "/api/product":
            barcode = str(payload.get("barcode", "")).strip()
            result = analyse_label(barcode)
        elif endpoint == "/api/chat":
            barcode = str(payload.get("barcode", "")).strip()
            message = str(payload.get("message", "")).strip()
            history = payload.get("history") or []
            if not isinstance(history, list):
                history = []
            recent_scans = payload.get("recent_scans") or []
            if not isinstance(recent_scans, list):
                recent_scans = []
            result = answer_question(barcode, message, history, recent_scans)
        else:
            image = str(payload.get("image", ""))
            _, separator, encoded_image = image.partition(",")
            if not separator or len(encoded_image) > 12_000_000:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid barcode image")
                return
            try:
                image_bytes = base64.b64decode(encoded_image, validate=True)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
                    temporary.write(image_bytes)
                    image_path = temporary.name
                try:
                    barcode, confidence = extract_text(image_path)
                finally:
                    Path(image_path).unlink(missing_ok=True)
            except (binascii.Error, OSError, ValueError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Could not decode barcode image")
                return
            if not barcode:
                barcode = decode_barcode_with_vision(image)
            if not barcode:
                barcode = decode_barcode_with_ocr(image_bytes)
            confidence = 1.0 if barcode else 0.0
            result = {"barcode": barcode, "confidence": confidence}

        body = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("", 8000), IngridHandler)
    print("Ingrid is running at http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()