"""Prepare figures, posters, three-view recordings and the captioned PowerPoint demo.

Run from the repository root. Requires ffmpeg, Pillow, and pypdfium2.
Original PowerPoint and experiment recordings are never overwritten.
"""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pypdfium2 as pdfium
from compose_views import compose_all
from export_ppt import main as export_ppt
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "web"
ASSETS = ROOT / "docs/website/assets"
WORK = ROOT / "playground/website/render"
TASKS = {
    "drawer": "Drawer packing",
    "gift": "Box packing and closing",
    "badminton": "Shuttlecock insertion",
    "cup": "Paper-cup stacking",
    "cube": "Block stacking",
}


def font(size, serif=False):
    name = "instrument-serif.ttf" if serif else "dm-sans-400.ttf"
    return ImageFont.truetype(str(ASSETS / "fonts" / name), size)


def posters():
    # Preserve source frames; the webpage assembles its cover with a CSS grid.
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            "20",
            "-i",
            str(SOURCES / "real-exp/drawer/FDM-VLM/third.mp4"),
            "-frames:v",
            "1",
            str(ASSETS / "cover-drawer.png"),
        ],
        check=True,
    )
    shutil.copy2(SOURCES / "real-task/gift/frame_001082.png", ASSETS / "cover-box.png")
    shutil.copy2(SOURCES / "real-task/badminton/frame_000175.png", ASSETS / "cover-shuttlecock.png")
    image = Image.new("RGB", (1600, 750), "#202c28")
    layout = [
        ("cube/frame_000424.png", (0, 0, 996, 750)),
        ("gift/frame_001082.png", (1004, 0, 596, 371)),
        ("cup/frame_001443.png", (1004, 379, 596, 371)),
    ]
    for source, (x, y, width, height) in layout:
        frame = Image.open(SOURCES / "real-task" / source)
        image.paste(ImageOps.fit(frame, (width, height)), (x, y))
    image.save(ASSETS / "demo-poster.webp", quality=90)
    social = Image.new("RGB", (1200, 630), "#f6f5f0")
    draw = ImageDraw.Draw(social)
    draw.text((55, 45), "Forward Dynamics Model", font=font(88, True), fill="#202c28")
    draw.text((58, 162), "Learn from the future. Act in the present.", font=font(25), fill="#25564c")
    social.paste(ImageOps.fit(image, (1090, 340)), (55, 245))
    social.save(ASSETS / "social.jpg", quality=90)


def main():
    ASSETS.mkdir(exist_ok=True)
    posters()
    WORK.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(ROOT / "ICLR27_Forwar_Dynamics_Models.zip")
    for name in ["archi_compare", "pipeline", "real_results_bar", "expset"]:
        data = archive.read(f"figure/{name}.pdf")
        (ASSETS / f"{name}.pdf").write_bytes(data)
        doc = pdfium.PdfDocument(data)
        doc[0].render(scale=2.5).to_pil().save(ASSETS / f"{name}.webp", quality=94)
    compose_all()
    export_ppt()


if __name__ == "__main__":
    main()
