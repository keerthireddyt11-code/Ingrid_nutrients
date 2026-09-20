import numpy as np
import pytest

import barcode_detector as bd


def test_extract_text_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.jpg"
    with pytest.raises(FileNotFoundError):
        bd.extract_text(str(missing))


def test_extract_text_raises_for_unreadable_file(tmp_path):
    bogus = tmp_path / "not-an-image.jpg"
    bogus.write_bytes(b"this is not image data")
    with pytest.raises(ValueError):
        bd.extract_text(str(bogus))


def test_extract_text_blank_image_has_no_barcode(tmp_path):
    import cv2

    blank = np.full((200, 300, 3), 255, dtype=np.uint8)
    image_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(image_path), blank)

    barcode, confidence = bd.extract_text(str(image_path))
    assert barcode == ""
    assert confidence == 0.0


def test_center_crops_returns_full_frame_plus_three_scaled_crops():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crops = bd._center_crops(image)

    assert len(crops) == 4
    assert crops[0] is image
    assert crops[1].shape == (80, 160, 3)
    assert crops[2].shape == (65, 130, 3)
    assert crops[3].shape == (50, 100, 3)


def test_center_crops_are_centred():
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crops = bd._center_crops(image)
    half_scale_crop = crops[3]
    expected = image[25:75, 25:75]
    assert np.array_equal(half_scale_crop, expected)
