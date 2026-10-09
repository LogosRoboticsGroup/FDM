"""Read optional per-frame subtask text from LeRobot parquet data."""


def attach_subtask(data: dict, dataframe, raw_step: int, config) -> None:
    key = getattr(config, "subtask_key", None)
    if key is None or key not in dataframe.columns:
        return
    value = dataframe.iloc[raw_step][key]
    if isinstance(value, str) and value.strip():
        data["subtask"] = value.strip()
