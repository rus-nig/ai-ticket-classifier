import json
from typing import List, Dict, Any

def parse_json_specifications(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array at the root.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")


def extract_key_information(specification: str) -> Dict[str, List[str]]:
    sections = {
        "user_story": [],
        "acceptance_criteria": [],
        "notes": []
    }

    lines = specification.splitlines()
    current_section = None
    for line in lines:
        line = line.strip()
        if line.lower().startswith("user story"):
            current_section = "user_story"
        elif line.lower().startswith("acceptance criteria"):
            current_section = "acceptance_criteria"
        elif line.lower().startswith("notes"):
            current_section = "notes"
        elif line and current_section:
            sections[current_section].append(line)
    return sections

