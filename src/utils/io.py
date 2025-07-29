import yaml
import json

from pathlib import Path
from typing import Union


def load_yaml(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(file_path: Union[str, Path], data) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True,
                  default_flow_style=False, indent=4)


def load_json(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: Union[str, Path], data) -> dict:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
