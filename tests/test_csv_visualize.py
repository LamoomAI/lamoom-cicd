from lamoom_cicd import TestLLMResponsePipe
import os
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())


def test_csv_compare():
    lamoom_pipe = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    results = lamoom_pipe.compare_from_csv("tests/test_data.csv")

    lamoom_pipe.visualize_test_results()

    assert 'score' in results[0].score.to_dict()
    assert 'passed' in results[0].score.to_dict()