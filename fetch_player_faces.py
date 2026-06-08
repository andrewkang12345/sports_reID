"""Download reference face photos for the 22 ARG/FRA 2022 WC Final players in the clip.

Source: Wikipedia/Wikimedia Commons. Each URL points to a head-on portrait that
InsightFace can embed. Saved to models/player_faces/<Team>_<Number>_<NameSlug>.jpg.

Usage:
    python3 fetch_player_faces.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path("models/player_faces")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (team, jersey, name, commons_image_url)
PLAYER_URLS: list[tuple[str, int, str, str]] = [
    # ----- Argentina -----
    ("Argentina", 10, "Lionel Messi",            "https://upload.wikimedia.org/wikipedia/commons/c/c1/Lionel_Messi_20180626.jpg"),
    ("Argentina", 7,  "Rodrigo De Paul",         "https://upload.wikimedia.org/wikipedia/commons/2/29/Rodrigo_de_Paul_2018.jpg"),
    ("Argentina", 9,  "Julian Alvarez",          "https://upload.wikimedia.org/wikipedia/commons/3/3c/Julian_Alvarez_Argentina_2022.jpg"),
    ("Argentina", 20, "Alexis Mac Allister",     "https://upload.wikimedia.org/wikipedia/commons/6/65/Alexis_Mac_Allister_2022.jpg"),
    ("Argentina", 24, "Enzo Fernandez",          "https://upload.wikimedia.org/wikipedia/commons/3/35/Enzo_Fernandez_Argentina_2022.jpg"),
    ("Argentina", 26, "Nahuel Molina",           "https://upload.wikimedia.org/wikipedia/commons/4/49/Molina-Argentina-Australia-WC-2022.jpg"),
    ("Argentina", 19, "Nicolas Otamendi",        "https://upload.wikimedia.org/wikipedia/commons/9/97/N._Otamendi_argentina.jpg"),
    ("Argentina", 13, "Cristian Romero",         "https://upload.wikimedia.org/wikipedia/commons/9/9c/Cristian_Romero_2022.jpg"),
    ("Argentina", 3,  "Nicolas Tagliafico",      "https://upload.wikimedia.org/wikipedia/commons/d/db/2018-08-08_FFF_-_AJAX_03.jpg"),
    ("Argentina", 11, "Angel Di Maria",          "https://upload.wikimedia.org/wikipedia/commons/4/40/%C3%81ngel_Di_Maria_2018.jpg"),
    ("Argentina", 23, "Emiliano Martinez",       "https://upload.wikimedia.org/wikipedia/commons/2/2c/Emiliano_Mart%C3%ADnez_2018.jpg"),
    # ----- France -----
    ("France",    10, "Kylian Mbappe",           "https://upload.wikimedia.org/wikipedia/commons/0/01/Kylian_Mbapp%C3%A9_2018.jpg"),
    ("France",    9,  "Olivier Giroud",          "https://upload.wikimedia.org/wikipedia/commons/0/0a/Olivier_Giroud_2018.jpg"),
    ("France",    7,  "Antoine Griezmann",       "https://upload.wikimedia.org/wikipedia/commons/2/2d/Antoine_Griezmann_2018.jpg"),
    ("France",    22, "Theo Hernandez",          "https://upload.wikimedia.org/wikipedia/commons/9/9c/Th%C3%A9o_Hern%C3%A1ndez_2018.jpg"),
    ("France",    14, "Adrien Rabiot",           "https://upload.wikimedia.org/wikipedia/commons/4/4d/Adrien_Rabiot_2018.jpg"),
    ("France",    18, "Dayot Upamecano",         "https://upload.wikimedia.org/wikipedia/commons/8/8f/Dayot_Upamecano_2019.jpg"),
    ("France",    4,  "Raphael Varane",          "https://upload.wikimedia.org/wikipedia/commons/c/c5/Raphael_Varane_2018.jpg"),
    ("France",    8,  "Aurelien Tchouameni",     "https://upload.wikimedia.org/wikipedia/commons/3/3c/Aur%C3%A9lien_Tchouam%C3%A9ni_2022.jpg"),
    ("France",    5,  "Jules Kounde",            "https://upload.wikimedia.org/wikipedia/commons/5/56/Jules_Kound%C3%A9_2021.jpg"),
    ("France",    11, "Ousmane Dembele",         "https://upload.wikimedia.org/wikipedia/commons/d/d5/Ousmane_Dembele_2018.jpg"),
]


def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def fetch(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 5_000:
        return True
    cmd = [
        "curl", "-sSL", "-A", "Mozilla/5.0 sports-reid",
        "--max-time", "30", "-o", str(dst), url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if dst.stat().st_size < 5_000:
            print(f"  [tiny file?] {dst.name} ({dst.stat().st_size} bytes)")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [fail] {url}: {e.stderr.decode()[:200]}")
        return False


def main() -> None:
    ok = fail = 0
    for team, jersey, name, url in PLAYER_URLS:
        dst = OUT_DIR / f"{team}_{jersey:02d}_{slugify(name)}.jpg"
        print(f"  {team} #{jersey:2d} {name} -> {dst.name}", end="")
        if fetch(url, dst):
            print(" ok")
            ok += 1
        else:
            print(" FAIL")
            fail += 1
        time.sleep(0.3)  # be polite to Wikimedia
    print(f"\nDone: {ok} ok, {fail} failed.  Files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
