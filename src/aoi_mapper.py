"""Local What/Where to AOI mapping for the project extension."""

from __future__ import annotations

AOI_MAPPING = {
    ("no", "signal"): "off_screen",
    ("screen", "other"): "screen_nonAOI",
    ("woman", "face"): "woman_face",
    ("man", "face"): "man_face",
    ("toy", "other"): "toy_present",
    ("toy2", "other"): "toy_location",
    ("man", "body"): "man_body",
    ("woman", "body"): "woman_body",
    ("man", "hands"): "man_hands",
    ("woman", "hands"): "woman_hands",
}


def map_what_where(what: str, where: str) -> str:
    """Map What/Where values to an AOI label."""
    key = (str(what).strip().lower(), str(where).strip().lower())
    try:
        return AOI_MAPPING[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported AOI combination: {key}") from exc


__all__ = ["AOI_MAPPING", "map_what_where"]

