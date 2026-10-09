"""Export the supplied PPT as timed slides plus its full embedded experiment videos.

Requires LibreOffice Impress, ffmpeg, and pypdfium2. Original PPT is never modified.
PowerPoint build animations become static slide compositions; embedded videos retain 1x speed.
"""

import json
import math
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs/website/assets"
WORK = ROOT / "playground/website/ppt-export"
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
SECONDS = [5, 7, 10, 7, 9, 10, 12, 10, 10, 12, 9, 10, 8]
CAPTIONS = [
    "Forward Dynamics Model: learn from the future, act in the present.",
    "FDM uses future prediction to improve action learning across two backbone families.",
    "VLA predicts actions directly. Future-conditioned WAM predicts futures first. FDM reverses this dependency.",
    "The generation branch reads action hidden states. Future-loss gradients supervise the action expert.",
    "Context and action features condition future prediction. Future information cannot flow back into action inputs.",
    "Mixture-of-transformers attention connects context, action, and generation through hidden states.",
    "Asymmetric attention blocks both direct and indirect future-information leakage into the action policy.",
    "With a video backbone, future flow-matching loss trains the action expert through its hidden states.",
    "With a vision-language backbone, FDM directly regresses future-image latents, using an MSE objective.",
    "The video instantiation predicts future video through iterative denoising with one-way access to action features.",
    "At inference, keep current-observation processing and action prediction. Future generation is optional.",
    "Both instantiations use future supervision during training while supporting action-only deployment.",
    "Five real-robot tasks, four methods, and three camera views. Recordings play at original speed.",
    "Drawer packing. The third view sits above both wrist views. Shorter recordings hold their final frame.",
    "Box packing and closing. Four methods perform the same task, shown at 1x speed.",
    "Shuttlecock insertion. The three camera views show each policy's execution from complementary angles.",
    "Paper-cup stacking. Compare grasping, placement, and task completion across the four methods.",
    "Block stacking. Individual recordings illustrate behavior; aggregate success rates are reported in the paper.",
]


def run(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def stamp(seconds, separator="."):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{separator}{ms % 1000:03d}"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    source = zipfile.ZipFile(ROOT / "web/iclr27-fdm-demo.pptx")
    if source.testzip() is not None:
        raise ValueError("PowerPoint ZIP integrity check failed")
    presentation = WORK / "presentation.pptx"
    with zipfile.ZipFile(presentation, "w", zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            data = source.read(name)
            if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                data = data.replace(b"Forward Dynamics Models", b"Forward Dynamics Model")
                data = data.replace(b"ICLR submission 3770", b"").replace(b"Act2Future: overview", b"FDM: overview")
                # Rasterize the poster only; ffmpeg overlays the actual video afterward.
                slide = ET.fromstring(data)
                for parent in slide.iter():
                    for child in list(parent):
                        if child.tag.rsplit("}", 1)[-1] in {"videoFile", "media", "timing"}:
                            parent.remove(child)
                data = ET.tostring(slide, encoding="utf-8", xml_declaration=True)

            target.writestr(name, data)
            if name.endswith(".mp4"):
                (WORK / Path(name).name).write_bytes(source.read(name))
    run(
        "libreoffice",
        "-env:UserInstallation=file:///tmp/fdm-lo-profile",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(WORK),
        str(presentation),
    )
    pdf = pdfium.PdfDocument(WORK / "presentation.pdf")
    for index, page in enumerate(pdf):
        page.render(scale=1920 / page.get_width()).to_pil().convert("RGB").save(WORK / f"slide-{index + 1:02d}.png")
    size = ET.fromstring(source.read("ppt/presentation.xml")).find("p:sldSz", NS)
    sx, sy = 1920 / int(size.attrib["cx"]), 1080 / int(size.attrib["cy"])
    durations = list(SECONDS)
    boxes = []
    for number in range(14, 19):
        slide = ET.fromstring(source.read(f"ppt/slides/slide{number}.xml"))
        picture = next(pic for pic in slide.findall(".//p:pic", NS) if pic.find(".//a:videoFile", NS) is not None)
        transform = picture.find("p:spPr/a:xfrm", NS)
        offset, extent = transform.find("a:off", NS), transform.find("a:ext", NS)
        boxes.append(
            (
                round(int(offset.attrib["x"]) * sx),
                round(int(offset.attrib["y"]) * sy),
                2 * round(int(extent.attrib["cx"]) * sx / 2),
                2 * round(int(extent.attrib["cy"]) * sy / 2),
            )
        )
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(WORK / f"media{number - 13}.mp4"),
                ]
            )
        )
        durations.append(math.ceil(float(info["format"]["duration"]) * 30) / 30)

    def encode(index):
        args = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(WORK / f"slide-{index + 1:02d}.png"),
        ]
        if index >= 13:
            x, y, width, height = boxes[index - 13]
            args += [
                "-i",
                str(WORK / f"media{index - 12}.mp4"),
                "-filter_complex_threads",
                "1",
                "-filter_complex",
                f"[1:v]scale={width}:{height},setsar=1[v];[0:v][v]overlay={x}:{y}[out]",
                "-map",
                "[out]",
            ]
        args += [
            "-t",
            str(durations[index]),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(WORK / f"part-{index:02d}.mp4"),
        ]
        run(*args)
        print(f"Rendered slide {index + 1}/18", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(encode, range(18)))
    concat = WORK / "concat.txt"
    concat.write_text("".join(f"file '{WORK / f'part-{index:02d}.mp4'}'\n" for index in range(18)))
    run(
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(ASSETS / "demo.mp4"),
    )
    cursor = 0
    cues = []
    for index, duration in enumerate(durations):
        end = cursor + (min(duration, 12) if index >= 13 else duration)
        cues.append((cursor, end, CAPTIONS[index]))
        cursor += duration
    (ASSETS / "demo.en.vtt").write_text(
        "WEBVTT\n\n" + "\n\n".join(f"{stamp(a)} --> {stamp(b)}\n{text}" for a, b, text in cues) + "\n"
    )
    (ASSETS / "demo.en.srt").write_text(
        "\n\n".join(f"{i + 1}\n{stamp(a, ',')} --> {stamp(b, ',')}\n{text}" for i, (a, b, text) in enumerate(cues))
        + "\n"
    )
    (ASSETS / "demo-transcript.txt").write_text(
        "Forward Dynamics Model — video transcript\n\n" + "\n\n".join(CAPTIONS) + "\n"
    )
    (ASSETS / "demo-metadata.json").write_text(
        json.dumps(
            {"duration": cursor, "slides": 18, "source": "iclr27-fdm-demo.pptx", "slide_durations": durations}, indent=2
        )
        + "\n"
    )
    run(
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(ASSETS / "demo.mp4"),
        "-vf",
        f"subtitles={ASSETS / 'demo.en.srt'}:force_style='FontSize=18,MarginV=26'",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-threads",
        "4",
        "-movflags",
        "+faststart",
        str(ASSETS / "demo-captioned.mp4"),
    )
    print(f"Exported {cursor:.2f}s PowerPoint demo.", flush=True)


if __name__ == "__main__":
    main()
