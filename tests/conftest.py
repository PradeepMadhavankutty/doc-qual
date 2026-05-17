"""Shared pytest fixtures for Doc-Qual tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def clean_doc() -> np.ndarray:
    """Synthetic document image: light background with dark text lines."""
    img = np.ones((300, 500), dtype=np.uint8) * 240
    for y in range(40, 260, 25):
        img[y : y + 6, 50:450] = 25
    return img
