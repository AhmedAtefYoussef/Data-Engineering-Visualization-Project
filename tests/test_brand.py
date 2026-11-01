"""The mark beside the title and the browser icon draw the same picture: buildings as bars on a street, and a crash
point inside its 250 m ring."""
from __future__ import annotations

import struct
from pathlib import Path

from app import page

ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"


def _classes(node):
    yield from str(getattr(node, "className", "") or "").split()
    for child in getattr(node, "children", None) or []:
        if not isinstance(child, str):
            yield from _classes(child)


def test_the_mark_has_three_bars_a_street_and_a_ring():
    names = list(_classes(page._brand_mark()))
    assert names.count("bm-bar") == 3
    assert {"bm-street", "bm-ring", "bm-pulse", "bm-dot"} <= set(names)


def test_the_icons_are_the_new_picture_at_every_size():
    svg = (ASSETS / "favicon.svg").read_text(encoding="utf-8")
    assert svg.count("<rect") >= 5 and "#ff8f78" in svg and "#D55E00" not in svg
    # Read from the file headers, so the test needs no imaging library: an ICO lists its images in 16-byte entries
    # after a 6-byte header; a PNG gives its width and height right after the IHDR tag.
    ico = (ASSETS / "favicon.ico").read_bytes()
    _, kind, count = struct.unpack("<HHH", ico[:6])
    sizes = {(ico[6 + 16 * k] or 256, ico[7 + 16 * k] or 256) for k in range(count)}
    assert kind == 1 and {(16, 16), (32, 32), (48, 48)} <= sizes
    png = (ASSETS / "apple-touch-icon.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (180, 180)


def test_the_page_head_links_both_icons():
    source = (ASSETS.parent / "app.py").read_text(encoding="utf-8")
    assert 'href="/assets/favicon.svg"' in source and 'href="/assets/apple-touch-icon.png"' in source
