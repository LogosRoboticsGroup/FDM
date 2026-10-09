"""Compose third (640x480), left/right wrist (320x240 each) into 640x720 videos."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "web"
ASSETS = ROOT / "docs/website/assets"
VIEWS = [("third", "Third view", 640, 480), ("left", "Left wrist", 320, 240), ("right", "Right wrist", 320, 240)]


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"]) if result.returncode == 0 else None


def compose(directory):
    key = directory.relative_to(SOURCES / "real-exp").as_posix()
    output = ASSETS / "comparisons" / f"{key}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = {view: probe(directory / f"{view}.mp4") for view, _, _, _ in VIEWS}
    duration = max(value for value in durations.values() if value is not None)
    inputs, filters = [], []
    font = ASSETS / "fonts/dm-sans-400.ttf"
    for index, (view, label, width, height) in enumerate(VIEWS):
        if durations[view] is None:
            inputs += ["-f", "lavfi", "-i", f"color=c=0x202c28:s={width}x{height}:r=30:d={duration}"]
            missing = (
                f",drawtext=fontfile={font}:text='Source incomplete':fontcolor=white:"
                f"fontsize=18:x=(w-tw)/2:y=(h-th)/2"
            )
        else:
            inputs += ["-threads", "1", "-i", str(directory / f"{view}.mp4")]
            missing = ""
        filters.append(
            f"[{index}:v]scale={width}:{height},setsar=1,fps=30,setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},"
            f"drawtext=fontfile={font}:text='{label}':fontcolor=white:fontsize=16:"
            f"x=10:y=10:box=1:boxcolor=0x202c28@0.8:boxborderw=5{missing}[v{index}]"
        )
    filters.extend(["[v1][v2]hstack=inputs=2[wrists]", "[v0][wrists]vstack=inputs=2[out]"])
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        *inputs,
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-t",
        str(duration),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(output),
            "-frames:v",
            "1",
            str(output.with_suffix(".webp")),
        ],
        check=True,
    )
    print(f"Composed {key}", flush=True)
    return f"{key}.mp4", {
        "duration": round(duration, 2),
        "missing_views": [v for v, d in durations.items() if d is None],
    }


def compose_all():
    directories = sorted({path.parent for path in (SOURCES / "real-exp").rglob("third.mp4")})
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = dict(pool.map(compose, directories))
    (ASSETS / "comparisons.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    compose_all()
