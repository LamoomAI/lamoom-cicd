from lamoom_cicd import TestLLMResponsePipe
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

def test_compare():
    ideal_answer = """Headaches can have many causes, including stress, dehydration, 
    lack of sleep, or more serious conditions like migraines or sinusitis. 
    It's important to keep track of when the headache started, how long it lasts, 
    and any other symptoms you have. If it's persistent or severe, you should 
    see a doctor for a proper diagnosis."""

    llm_response = """
    Headaches can stem from various factors such as stress, dehydration, insufficient sleep, 
    or even more serious conditions like migraines or sinusitis. Keeping track of when 
    the headache begins, its duration, and any accompanying symptoms is crucial. 
    If the headache is persistent or severe, it's best to consult your doctor 
    for a proper diagnosis."""    
    
    optional_params =  {'prompt_id': "medical_help"}

    lamoom_pipe = TestLLMResponsePipe(openai_key=os.environ.get("OPENAI_KEY"))
    result = lamoom_pipe.compare(ideal_answer, llm_response, optional_params=optional_params)

    lamoom_pipe.visualize_test_results()
    
    assert 'score' in result.score.to_dict()
    assert 'passed' in result.score.to_dict()