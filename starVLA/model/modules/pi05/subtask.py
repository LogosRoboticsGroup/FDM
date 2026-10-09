"""Resolve optional frame annotations for joint language/action supervision."""

from collections.abc import Mapping

SUBTASK_KEYS = ("subtask", "subtask_generation", "distribute", "subtask_generation_zh", "distribute_zh")


def resolve_subtask(example: dict) -> str | None:
    info = example.get("instruction_info", example)
    for key in SUBTASK_KEYS:
        value = info.get(key)
        if isinstance(value, Mapping):
            frame = int(example["frame_index"])
            value = next(
                (text for span, text in value.items() if _contains_frame(span, frame)),
                None,
            )
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _contains_frame(span: str, frame: int) -> bool:
    start, end = map(int, span.split())
    return start <= frame < end or frame == start
