import json

import cv2
import numpy as np

from doc_qual.cli import main


def make_clean_doc() -> np.ndarray:
    img = np.ones((180, 280), dtype=np.uint8) * 240
    for y in range(30, 155, 25):
        img[y : y + 5, 30:250] = 25
    return img


def test_cli_json_output(tmp_path, capsys) -> None:
    image_path = tmp_path / "doc.png"
    cv2.imwrite(str(image_path), make_clean_doc())
    exit_code = main([str(image_path), "--format", "json", "--threshold", "0"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert "ocr_score" in payload
    assert payload["passed"] is True
