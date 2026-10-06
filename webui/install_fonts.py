"""Build the free-for-commercial-use subtitle font library (fonts/free).

    python webui/install_fonts.py            # packages in fonts/*.zip + English fonts from Google Fonts
    python webui/install_fonts.py --no-english

One common weight of every package is extracted with its licence files; every
font is measured for the scripts it covers and listed in fonts/free/catalog.json.
Translated-video subtitles use only these fonts.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subalign.fontlib import FONTS_DIR, install  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-english", action="store_true", help="do not download English fonts")
    a = ap.parse_args()
    print(f"fonts: {FONTS_DIR}")
    entries = install(english=not a.no_english)
    unverified = [e.display for e in entries if not e.verified]
    print(f"{len(entries)} fonts -> {FONTS_DIR / 'free' / 'catalog.json'}")
    if unverified:
        print("only the download site's statement, no original licence included - check before use:", "、".join(unverified))
