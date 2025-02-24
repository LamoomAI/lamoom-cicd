from lamoom_cicd import TestLLMResponsePipe
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

    llm_response = """Blockchain is like a shared digital notebook where everyone has a copy.
    New records (blocks) are added in order and can’t be changed or erased. 
    Each block is securely locked with a code, and everyone in the network must agree 
    before adding new information. This makes blockchain transparent, secure, and
    tamper-proof, which is why it's used for things like cryptocurrency, secure transactions, 
    and digital contracts."""    
    
    optional_params =  {'prompt_id': "blockchain_prompt"}

    lamoom_pipe = TestLLMResponsePipe(openai_key=os.environ.get("OPENAI_KEY"))
    result = lamoom_pipe.compare(ideal_answer, llm_response, optional_params=optional_params)

    lamoom_pipe.visualize_test_results()
    
    assert 'score' in result.score.to_dict()
    assert 'passed' in result.score.to_dict()