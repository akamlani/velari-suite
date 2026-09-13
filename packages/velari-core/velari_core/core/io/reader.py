import  json
import  base64
import  pandas as pd
from    pathlib import Path
from    typing import Optional, Any, Union


# type="base64", media_type="application/pdf"
# type="text"
def read_base64(path: str) -> str:
    try:
        with Path(path).open("rb") as file:
            data = base64.standard_b64encode(file.read()).decode("utf-8")
            return data
    except Exception as e:
        raise IOError(f"Error reading file {path}: {e}")

def read_csv(path: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
    try:
        sep = kwargs.get("sep", ",")  # e.g., ['\s+', '\t', ',']
        with Path(path).open("r", encoding="utf-8") as file:
            result = pd.read_csv(file, *args, sep=sep, **kwargs)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("read_csv() only supports whole-file reads — pass iterator=False or omit chunksize")
        return result
    except Exception as e:
        raise IOError(f"Error reading CSV file {path}: {e}")

def read_excel(path: str, sheet_name: Union[str, int] = 0, *args: Any, **kwargs: Any) -> pd.DataFrame:
    try:
        with pd.ExcelFile(path) as reader:
            kwargs.setdefault("engine", "openpyxl")
            return pd.read_excel(reader, sheet_name=sheet_name, *args, **kwargs)
    except Exception as e:
        raise IOError(f"Error reading Excel file {path}: {e}")

def read_json(path: str) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        raise IOError(f"Error reading JSON file {path}: {e}")

def read_json_string(data: str) -> dict:
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        return {"__error__": str(e)}

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