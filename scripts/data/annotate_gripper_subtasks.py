"""Annotate subtasks using measured gripper opening (LeRobot v2).

Currently supports the red/yellow cube-stacking task templates.
Dry-run by default. --apply adds subtask_generation in place after backing up
parquet files and info.json. Frame intervals are half-open [start, end).
"""

import argparse
import json
import re
import shutil
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SUBTASK_KEY = "subtask_generation"


def gripper_events(values, arm, fps):
    """Use hysteresis; exclude initial closure and brief corrective squeezes."""
    events, ignored = [], []
    start = 0 if values[0] < 0.5 else None
    for frame in range(1, len(values)):
        if start is None:
            if values[frame] < 0.5:
                start = frame
            continue
        if values[frame] <= 0.8:
            continue
        event = {"arm": arm, "closed_frame": start, "open_frame": frame}
        if start == 0:
            ignored.append({**event, "reason": "initially_closed"})
        elif frame - start < int(np.ceil(0.6 * fps)):
            ignored.append({**event, "reason": "short_closure"})
        else:
            # The median held opening adapts to different cube/gripper widths.
            held_opening = float(np.median(values[start:frame]))
            threshold = held_opening + 0.1 * (0.95 - held_opening)
            settled = values[start:frame] <= threshold
            stable = np.flatnonzero(np.convolve(settled.astype(int), np.ones(3, dtype=int), "valid") == 3)
            previous_open = np.flatnonzero(values[:start] >= 0.95)
            release = np.flatnonzero(values[frame:] >= 0.95)
            if not len(stable) or not len(previous_open):
                raise ValueError(f"{arm}: incomplete grasp around frame {start}")
            events.append(
                {
                    **event,
                    "close_start": int(previous_open[-1]) + 1,
                    "close_complete": start + int(stable[0]) + 2,
                    "release_complete": frame + int(release[0]) if len(release) else None,
                    "held_opening": held_opening,
                    "settled_threshold": threshold,
                }
            )
        start = None
    if start is not None:
        if start == 0:
            ignored.append({"arm": arm, "closed_frame": 0, "open_frame": None, "reason": "initially_closed"})
        else:
            raise ValueError(f"{arm}: unreleased closure at frame {start}")
    return events, ignored


def annotate_episode(table, task, fps, gripper_indices):
    match = re.fullmatch(r"Stack the two cubes, first (red|yellow), then (red|yellow)", task)
    if match is None or match[1] == match[2]:
        raise ValueError(f"Unsupported task: {task}")
    state = np.asarray(table["state.joint"].to_pylist())
    events, ignored = [], []
    for arm, index in gripper_indices.items():
        accepted, rejected = gripper_events(state[:, index], arm, fps)
        events.extend(accepted)
        ignored.extend(rejected)
    events.sort(key=lambda event: event["close_start"])
    if len(events) != 2:
        raise ValueError(f"Expected two sustained grasps, found {len(events)}: {events}")
    first, second = match.groups()
    texts = [
        f"Move to the {first} cube",
        f"Pick up the {first} cube",
        "Stack on the wood cube",
        f"Move to the {second} cube",
        f"Pick up the {second} cube",
        f"Stack on the {first} cube",
    ]
    a, b = events
    if a["release_complete"] is None:
        raise ValueError("First grasp was not fully released")
    bounds = [
        0,
        a["close_start"],
        a["close_complete"],
        a["release_complete"],
        b["close_start"],
        b["close_complete"],
        len(state),
    ]
    if any(end <= start for start, end in pairwise(bounds)):
        raise ValueError(f"Empty or overlapping subtask intervals: {bounds}")
    segments = [
        {"start": start, "end": end, "subtask": text}
        for start, end, text in zip(bounds[:-1], bounds[1:], texts, strict=True)
    ]
    warnings = ["final_release_below_0.95"] if b["release_complete"] is None else []
    return {
        "task": task,
        "length": len(state),
        "events": events,
        "ignored_events": ignored,
        "warnings": warnings,
        "segments": segments,
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.report_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    tasks = {
        row["task_index"]: row["task"] for row in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())
    }
    names = info["features"]["state.joint"]["names"]
    indices = {arm: names.index(f"{arm}.gripper_open") for arm in ("left", "right")}
    records, errors, paths = [], [], []
    for path in sorted(root.glob("data/*/*.parquet")):
        table = pq.read_table(path)
        episode = int(table["episode_index"][0].as_py())
        try:
            if SUBTASK_KEY in table.column_names:
                raise ValueError("Existing subtask annotations; refusing to replace them")
            if len(set(table["task_index"].to_pylist())) != 1:
                raise ValueError("Episode has multiple task indices")
            if table["frame_index"].to_pylist() != list(range(len(table))):
                raise ValueError("Frame indices are not contiguous and zero-based")
            record = annotate_episode(table, tasks[table["task_index"][0].as_py()], info["fps"], indices)
            records.append({"episode_index": episode, **record})
            paths.append(path)
        except ValueError as error:
            errors.append({"episode_index": episode, "error": str(error)})
    write_jsonl(output / "subtasks.jsonl", records)
    write_jsonl(output / "errors.jsonl", errors)
    summary = {
        "dataset": str(root),
        "episodes": len(records),
        "frames": sum(record["length"] for record in records),
        "errors": errors,
        "tasks": dict(Counter(record["task"] for record in records)),
        "ignored_events": dict(Counter(event["reason"] for record in records for event in record["ignored_events"])),
        "flagged_episodes": [record["episode_index"] for record in records if record["ignored_events"]],
        "warnings": [
            {"episode_index": record["episode_index"], "warnings": record["warnings"]}
            for record in records
            if record["warnings"]
        ],
        "rules": {
            "signal": "state.joint left.gripper_open + right.gripper_open",
            "closed_threshold": 0.5,
            "reopen_threshold": 0.8,
            "fully_open_threshold": 0.95,
            "minimum_hold_seconds": 0.6,
            "pick_end": "3 frames at <= held median + 0.1 * (0.95 - held median)",
            "move_end": "first frame below 0.95 following last fully-open frame before sustained closure",
            "first_stack_end": "first frame >= 0.95 after first release",
            "final_stack_end": "episode end, including retreat frames",
            "intervals": "[start, end)",
            "caveat": "Gripper-based pseudo-labels; do not establish grasp/stack success or exact lift timing.",
        },
        "applied": False,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if errors or len(records) != info["total_episodes"] or summary["frames"] != info["total_frames"]:
        raise RuntimeError(f"Dataset validation failed; inspect {summary_path}")
    if args.apply:
        if SUBTASK_KEY in info["features"]:
            raise ValueError(f"{SUBTASK_KEY} already exists; refusing to replace existing annotations")
        backup = output / "original"
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        # Back up all files before the first dataset mutation.
        for path in [info_path, *paths]:
            target = backup / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for path, record in zip(paths, records, strict=True):
            table = pq.read_table(path)
            if SUBTASK_KEY in table.column_names:
                raise ValueError(f"Existing annotation column in {path}")
            labels = [
                segment["subtask"] for segment in record["segments"] for _ in range(segment["start"], segment["end"])
            ]
            table = table.append_column(SUBTASK_KEY, pa.array(labels, type=pa.string()))
            temp = path.with_suffix(".parquet.tmp")
            pq.write_table(table, temp)
            temp.replace(path)
        info["features"][SUBTASK_KEY] = {"dtype": "string", "shape": [1], "names": None}
        temp = info_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
        temp.replace(info_path)
        shutil.copy2(output / "subtasks.jsonl", root / "meta/subtasks.jsonl")
        summary.update(applied=True, backup=str(backup))
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        shutil.copy2(summary_path, root / "meta/subtask_annotation.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
