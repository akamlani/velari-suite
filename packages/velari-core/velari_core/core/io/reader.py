import json
import base64
from   pathlib import Path


# type="base64", media_type="application/pdf"
# type="text"
def read_base64(path: str) -> str:
    try:
        with Path(path).open("rb") as file:
            data = base64.standard_b64encode(file.read()).decode("utf-8")
            return data
    except Exception as e:
        raise IOError(f"Error reading file {path}: {e}")

def read_json(path: str) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        raise IOError(f"Error reading JSON file {path}: {e}")

def read_text(path: str) -> str:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = file.read()
            return data
    except Exception as e:
        raise IOError(f"Error reading text file {path}: {e}")