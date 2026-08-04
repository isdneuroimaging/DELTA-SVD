#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render the Open Graph preview card for the documentation site.

Writes 'docs/assets/og-image.png', the image that 'overrides/main.html' points
link-preview scrapers at, from the pipeline figure on the docs landing page. The
output is committed; the docs build only copies it. Re-run after changing that
figure, the tagline, or the card design.

It is a maintainer tool and lives outside the container image, so it runs on the
Python it is invoked with and needs Pillow there. The card is set in Roboto to
match the docs theme, falling back to DejaVu Sans if Roboto is not installed;
'OG_FONT' points it at a Roboto variable font that is not installed system-wide:

    OG_FONT=~/fonts/'Roboto[wdth,wght].ttf' tools/make_og_image.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "assets" / "delta-svd.webp"
TARGET = ROOT / "docs" / "assets" / "og-image.png"

# 1200x630 is the size every major scraper renders without recropping.
WIDTH, HEIGHT = 1200, 630
MARGIN = 72

TITLE = "DELTA-SVD"
TAGLINE = [
    "Diffusion Endpoints for Longitudinal Tracking",
    "of white matter Alterations in cerebral Small Vessel Disease",
]
FOOTER = "delta-svd.com"

# Sampled from the figure's FW and MD maps, for the accent rule under the title.
CYAN = (0, 158, 219)
ORANGE = (222, 108, 20)

# The band, in pixels of the 2836x1584 figure: the timepoint-1 row from fwc-FA
# rightwards, which holds only maps and histograms. Starting further left pulls
# in the "Free Water Imaging" arrow, whose text is illegible at card width.
CROP = (610, 66, 2815, 445)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
# Roboto ships as one variable font covering every weight the card uses.
FONT_VARIABLE = [
    os.environ.get("OG_FONT", ""),
    "/usr/share/fonts/truetype/roboto/Roboto[wdth,wght].ttf",
    str(Path.home() / ".local/share/fonts/Roboto[wdth,wght].ttf"),
]


def load_font(size: int, weight: int) -> ImageFont.FreeTypeFont:
    """Return Roboto at the given size and weight, or the best fallback."""
    for path in FONT_VARIABLE:
        if not path or not Path(path).expanduser().is_file():
            continue
        font = ImageFont.truetype(str(Path(path).expanduser()), size)
        try:
            # Roboto declares its axes in the order (wght, wdth).
            font.set_variation_by_axes([weight, 100])
            return font
        except OSError:
            break  # Pillow built without variable-font support; fall through.
    candidates = FONT_CANDIDATES_BOLD if weight >= 600 else FONT_CANDIDATES
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise SystemExit(f"none of these fonts are installed: {candidates}")


def build() -> Image.Image:
    """Compose the card: figure band at the bottom, text above it."""
    card = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))

    # The brain maps, full card width so they bleed off both edges.
    band = Image.open(SOURCE).convert("RGB").crop(CROP)
    band_height = round(band.height * WIDTH / band.width)
    band = band.resize((WIDTH, band_height), Image.LANCZOS)

    # Ramp the top third to transparent, dissolving the seam against the card.
    fade = Image.linear_gradient("L").resize((WIDTH, band_height))
    fade = fade.point(lambda v: min(255, round(v * 3)))
    card.paste(band, (0, HEIGHT - band_height), fade)

    draw = ImageDraw.Draw(card)
    y = MARGIN

    draw.text((MARGIN, y), TITLE, font=load_font(92, 700), fill=(255, 255, 255))
    y += 118

    for x in range(180):  # Accent rule, cyan into orange.
        t = x / 179
        colour = tuple(round(a + (b - a) * t) for a, b in zip(CYAN, ORANGE))
        draw.rectangle((MARGIN + x, y, MARGIN + x, y + 5), fill=colour)
    y += 40

    tagline_font = load_font(31, 400)
    for line in TAGLINE:
        draw.text((MARGIN, y), line, font=tagline_font, fill=(190, 194, 201))
        y += 44

    # Below the tagline, not at the foot: the band would sit behind it there.
    draw.text((MARGIN, y + 26), FOOTER, font=load_font(26, 500), fill=(130, 136, 145))
    return card


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"figure not found: {SOURCE}")
    build().save(TARGET, "PNG", optimize=True)
    print(f"wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
