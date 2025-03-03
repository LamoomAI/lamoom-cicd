import os
import json
import pytest
from lamoom_cicd import TestLLMResponsePipe
from lamoom_cicd.responses import  TestCase, AggregatedResult

# Sample data for tests
IDEAL_ANSWER = """Headaches can have many causes, including stress, dehydration, 
    lack of sleep, or more serious conditions like migraines or sinusitis. 
    It's important to keep track of when the headache started, how long it lasts, 
    and any other symptoms you have. If it's persistent or severe, you should 
    see a doctor for a proper diagnosis.
"""

LLM_RESPONSES = [
    "Blockchain is a distributed ledger technology where changes are visible to all participants and can't be modified once recorded.",
    "Think of blockchain as a public, unchangeable record book. Once something is written, it stays there permanently for everyone to see.",
    "Blockchain is like a transparent database that's maintained by a network of computers rather than a single authority.",
    "Blockchain technology creates a tamper-proof record of transactions that doesn't require a central authority to verify or manage it.",
    "Blockchain is essentially a chain of blocks containing information that can't be altered once added, making it secure and transparent."
]

def test_save_and_load_test_case(tmp_path):
    """Test saving and loading test cases to/from JSON"""
    # Create a test case
    pipe = TestLLMResponsePipe()
    result = pipe.compare(IDEAL_ANSWER, LLM_RESPONSES[0], {"prompt_id": "blockchain_test"})
    
    # Save the test case
    test_case = pipe.save_test_case(result)
    case_id = test_case.case_id
    
    # Save to JSON
    json_path = os.path.join(tmp_path, "test_cases.json")
    pipe.save_test_cases_to_json(json_path)
    
    # Create a new pipe and load the test cases
    new_pipe = TestLLMResponsePipe()
    loaded_cases = new_pipe.load_test_cases_from_json(json_path)
    
    # Verify the loaded test case
    assert len(loaded_cases) == 1
    assert loaded_cases[0].case_id == case_id
    assert loaded_cases[0].prompt_id == "blockchain_test"
    assert loaded_cases[0].ideal_response == IDEAL_ANSWER
    assert len(loaded_cases[0].test_questions) > 0

def test_multiple_runs_and_aggregation(tmp_path):
    """Test running multiple tests with the same test case and aggregating results"""
    # Create a test case
    pipe = TestLLMResponsePipe()
    result = pipe.compare(IDEAL_ANSWER, LLM_RESPONSES[0], {"prompt_id": "blockchain_test"})
    test_case = pipe.save_test_case(result)
    case_id = test_case.case_id
    
    # Run multiple tests with different responses
    results = pipe.run_multiple_tests(case_id, LLM_RESPONSES)
    
    # Verify we have multiple results
    assert len(results) == len(LLM_RESPONSES)
    
    # Aggregate the results
    aggregated = pipe.aggregate_results(case_id)
    
    # Verify aggregation
    assert isinstance(aggregated, AggregatedResult)
    assert aggregated.case_id == case_id
    assert aggregated.runs == len(LLM_RESPONSES)
    assert 0 <= aggregated.avg_score <= 100
    assert len(aggregated.scores) == len(LLM_RESPONSES)
    
    # Save results
    results_path = os.path.join(tmp_path, "test_results.json")
    pipe.save_test_results_to_json(results_path)
    
    # Verify saved results
    assert os.path.exists(results_path)
    with open(results_path, 'r') as f:
        data = json.load(f)
        assert len(data) >= len(LLM_RESPONSES)

def test_save_test_cases_from_csv(tmp_path):
    """Test generating and saving test cases from a CSV file"""
    # Create a sample CSV file
    csv_path = os.path.join(tmp_path, "test_data.csv")
    with open(csv_path, 'w') as f:
        f.write("ideal_answer,llm_response,optional_params\n")
        f.write(f'"{IDEAL_ANSWER}","dummy response","{{\\"prompt_id\\": \\"blockchain_csv_test\\"}}"\n')
    
    # Generate test cases from CSV
    pipe = TestLLMResponsePipe()
    output_path = os.path.join(tmp_path, "generated_cases.json")
    cases = pipe.save_test_cases_from_csv(csv_path, output_path)
    
    # Verify generated cases
    assert len(cases) == 1
    assert cases[0].prompt_id == "blockchain_csv_test"
    assert cases[0].ideal_response == IDEAL_ANSWER
    assert len(cases[0].test_questions) > 0
    
    # Verify saved JSON
    assert os.path.exists(output_path)
    with open(output_path, 'r') as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["prompt_id"] == "blockchain_csv_test"