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


def process_all_specifications(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed_data = []
    for spec in data:
        specification_text = spec.get("specification", "")
        key_info = extract_key_information(specification_text)
        processed_data.append({
            "original_specification": specification_text,
            "key_information": key_info,
            "test_cases": spec.get("test_cases", []),
            "gherkin": spec.get("gherkin", [])
        })
    return processed_data


if __name__ == "__main__":
    # Example usage
    example_file = "data/example_data.json"

    try:
        all_specifications = parse_json_specifications(example_file)

        processed = process_all_specifications(all_specifications)

        for idx, item in enumerate(processed, start=1):
            print(f"Specification {idx}:")
            print("Original Specification:")
            print(item["original_specification"])
            print("\nExtracted Key Information:")
            print(item["key_information"])
            print("\nTest Cases:")
            for tc in item["test_cases"]:
                print(f"- Description: {tc['description']}")
                print(f"  Steps: {', '.join(tc['steps'])}")
                print(f"  Expected Result: {tc['expected_result']}")
            print("\nGherkin Scenarios:")
            for scenario in item["gherkin"]:
                print(f"- {scenario}")
            print("\n" + "="*40 + "\n")

    except Exception as e:
        print(f"Error: {e}")