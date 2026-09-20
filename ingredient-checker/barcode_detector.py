from pathlib import Path

import cv2


barcode_detector = cv2.barcode.BarcodeDetector()


def _has_valid_barcode_checksum(value):
    """Validate UPC/EAN check digits for automatically detected candidates."""
    digits = str(value or "").strip()
    if len(digits) not in {12, 13, 14} or not digits.isdigit():
        return False
    total = 0
    for index, digit in enumerate(reversed(digits[:-1])):
        total += int(digit) * (3 if index % 2 == 0 else 1)
    return (10 - total % 10) % 10 == int(digits[-1])


def _center_crops(image):
    """Return full-frame and centred crops for barcodes that occupy a small area."""
    height, width = image.shape[:2]
    crops = [image]
    for scale in (0.8, 0.65, 0.5):
        crop_height = int(height * scale)
        crop_width = int(width * scale)
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        crops.append(image[top:top + crop_height, left:left + crop_width])
    return crops


def _decode_barcode(image):
    """Try to decode one barcode from an image or a preprocessed variant."""
    variants = []
    for crop in _center_crops(image):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        prepared = [
            crop,
            cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ]
        for variant in prepared:
            variants.extend(
            [
                variant,
                cv2.rotate(variant, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(variant, cv2.ROTATE_90_COUNTERCLOCKWISE),
            ]
            )

    for variant in variants:
        try:
            decoded, _, _ = barcode_detector.detectAndDecode(variant)
        except cv2.error:
            decoded = ""
        if decoded and decoded.strip():
            return decoded.strip()

        try:
            detected, decoded_values, _, _ = barcode_detector.detectAndDecodeMulti(variant)
        except (AttributeError, cv2.error):
            continue
        if detected:
            for value in decoded_values:
                if value and value.strip():
                    return value.strip()

    return ""


def extract_text(image_path: str) -> tuple[str, float]:
    """Scan an image for a product barcode and return its value and confidence."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    barcode = _decode_barcode(image)
    if len(barcode) == 8 or not _has_valid_barcode_checksum(barcode):
        barcode = ""
    return barcode, 1.0 if barcode else 0.0
