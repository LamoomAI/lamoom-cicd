import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

def parse_csv_file(file_path: str) -> list:
    """
    Reads a CSV file and returns a list of dictionaries with keys:
    - ideal_answer
    - llm_response
    - optional_params (parsed as dict if not empty)
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []

    test_cases = []
    for _, row in df.iterrows():
        case = {
            "ideal_answer": row.get("ideal_answer"),
            "llm_response": row.get("llm_response")
        }
        opt_params = row.get("optional_params")
        if pd.notna(opt_params) and opt_params:
            try:
                case["optional_params"] = json.loads(opt_params)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing optional_params: {e}")
                case["optional_params"] = None
        else:
            case["optional_params"] = None
        test_cases.append(case)

    return test_cases