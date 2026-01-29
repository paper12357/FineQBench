import json
import pandas as pd
import io
import contextlib
import numpy as np


def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, (np.ndarray, pd.Series)):
        return [to_python(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [
            {k: to_python(v) for k, v in row.items()}
            for row in obj.to_dict(orient="records")
        ]

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]

    return obj


class CodeManager:
    def __init__(self):
        pass

    def run_code(self, abs_path: str, code: str):
        if abs_path.endswith(".json"):
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif abs_path.endswith(".csv"):
            data = pd.read_csv(abs_path, comment="#")
        elif abs_path.endswith(".txt"):
            with open(abs_path, "r", encoding="utf-8") as f:
                data = f.read()
        else:
            raise ValueError("only support .json, .csv, .txt files")

        local_env = {"data": data}

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, {}, local_env)

        if "result" not in local_env:
            raise ValueError("user code must define a variable named 'result'")

        return to_python(local_env["result"])