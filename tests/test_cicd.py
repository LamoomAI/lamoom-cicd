from lamoom_cicd import TestLLMResponse
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

def test_compare():
    ideal_answer = """Blockchain is like a digital notebook that everyone can see 
            but no one can secretly change. Imagine a shared Google Doc where every change 
            is recorded forever, and no one can erase or edit past entries. 
            Instead of one company controlling it, thousands of computers around 
            the world keep copies, making it nearly impossible to hack or fake. 
            This is why it’s used for things like Bitcoin—to keep transactions 
            secure and transparent without needing a bank in the middle."""
    optional_params =  {'prompt': "Explain the concept of blockchain to someone with no technical background."}

    test_response = TestLLMResponse(openai_key=os.environ.get("OPENAI_KEY"))
    result = test_response.compare(ideal_answer, optional_params=optional_params)

    assert 'score' in result.score.to_dict()
    assert 'passed' in result.score.to_dict()