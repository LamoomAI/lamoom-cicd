import pandas as pd
import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Union, TypeVar, Type

logger = logging.getLogger(__name__)

T = TypeVar('T')

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

def save_to_json_file(file_path: str, data: Union[List[Dict], Dict]) -> bool:
    """
    Saves data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save (should be JSON serializable)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving to JSON file: {e}")
        return False

def load_from_json_file(file_path: str) -> Optional[Union[List[Dict], Dict]]:
    """
    Loads data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        The loaded data or None if there was an error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading from JSON file: {e}")
        return None

def load_objects_from_json(file_path: str, cls: Type[T]) -> List[T]:
    """
    Loads a list of objects from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        cls: Class to deserialize into
        
    Returns:
        List of objects of type cls
    """
    data = load_from_json_file(file_path)
    if not data:
        return []
        
    if not isinstance(data, list):
        data = [data]
        
    return [cls.from_dict(item) for item in data]

def generate_case_id() -> str:
    """Generate a unique test case ID"""
    return str(uuid.uuid4())