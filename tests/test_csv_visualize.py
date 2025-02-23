from lamoom_cicd import TestLLMResponse
import os
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())


def test_csv_compare():
    test_response = TestLLMResponse(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    results = test_response.compare_from_csv("tests/test_data.csv")

    test_response.visualize_test_results()

    assert 'score' in results[0].score.to_dict()
    assert 'passed' in results[0].score.to_dict()