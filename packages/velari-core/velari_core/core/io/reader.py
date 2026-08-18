import  json
import  base64
from    pathlib import Path
from    typing import Optional


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

def trsfrm_data_to_json(data: dict, filter: Optional[str] = None) -> str:
    try:
        if filter:
            filtered_data = {k: v for k, v in data.items() if filter in k}
            return json.dumps(filtered_data, sort_keys=True)
        else:
            return json.dumps(data, indent=2, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Error transforming data to JSON: {e}")