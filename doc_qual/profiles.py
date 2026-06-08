"""Engine-specific weight profile loader.

Bundled profiles ship as JSON files inside ``doc_qual/profiles/``.
They encode empirically-informed feature weights for each supported OCR
engine.  Users can also supply a path to a custom JSON file.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from pathlib import Path

ENGINE_PROFILES: tuple[str, ...] = ("tesseract", "textract", "azure", "paddleocr")

_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "sharpness",
        "noise",
        "edges",
        "skew",
        "brightness",
        "ridges",
    }
)


def load_engine_profile(engine: str) -> dict[str, float]:
    """Load weight profile for a named engine or a custom JSON path.

    Args:
        engine: One of ``'tesseract'``, ``'textract'``, ``'azure'``,
            ``'paddleocr'``, or an absolute/relative path to a custom
            JSON file containing a ``{feature: weight}`` mapping.

    Returns:
        Weight dict keyed by feature name.  Comment keys (``_comment``)
        are stripped.  Weights are **not** normalised here — the scorer
        normalises by the sum of weights present in the result, so
        profiles may omit features they don't use.

    Raises:
        ValueError: if the engine name is not recognised and the path
            does not exist.
        ValueError: if the loaded JSON is missing required feature keys.
    """
    path = Path(engine)
    if path.exists() and path.suffix == ".json":
        return _load_and_validate(path)

    if engine not in ENGINE_PROFILES:
        raise ValueError(
            f"Unknown engine '{engine}'. "
            f"Built-in profiles: {ENGINE_PROFILES}. "
            "Or pass the path to a custom JSON weight file."
        )

    # Load from package data using importlib.resources (works for editable
    # and wheel installs alike).
    try:
        pkg_path = pkg_resources.files("doc_qual") / "profiles" / f"{engine}.json"
        raw = pkg_path.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError) as exc:
        raise FileNotFoundError(
            f"Bundled profile '{engine}.json' not found inside the doc_qual package. "
            "Re-install the package to restore bundled profiles."
        ) from exc

    data = json.loads(raw)
    return _strip_and_validate(data, engine)


def load_custom_profile(path: str | Path) -> dict[str, float]:
    """Load and validate a weight profile from a user-supplied JSON file.

    Args:
        path: Path to a JSON file with ``{feature_name: weight}`` mapping.

    Returns:
        Validated weight dict (comment keys stripped).
    """
    return _load_and_validate(Path(path))


# ── internal helpers ────────────────────────────────────────────────────────


def _load_and_validate(path: Path) -> dict[str, float]:
    if not path.exists():
        raise ValueError(f"Profile file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _strip_and_validate(data, str(path))


def _strip_and_validate(data: dict, source: str) -> dict[str, float]:
    weights = {k: float(v) for k, v in data.items() if not k.startswith("_")}
    missing = _REQUIRED_KEYS - weights.keys()
    if missing:
        raise ValueError(
            f"Profile '{source}' is missing required feature keys: {sorted(missing)}"
        )
    negatives = {k: v for k, v in weights.items() if v < 0}
    if negatives:
        raise ValueError(f"Profile '{source}' has negative weights: {negatives}")
    return weights
