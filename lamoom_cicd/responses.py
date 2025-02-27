from dataclasses import dataclass

class Question:
    def __init__(self, test_question: str, llm_answer: str, ideal_answer: str, does_match_ideal_answer: bool):
        self.test_question = test_question
        self.llm_answer = llm_answer
        self.ideal_answer = ideal_answer
        self.does_match_ideal_answer = does_match_ideal_answer

    def to_dict(self):
        return {
            "test_question": self.test_question,
            "llm_answer": self.llm_answer,
            "ideal_answer": self.ideal_answer,
            "does_match_ideal_answer": self.does_match_ideal_answer
        }
        
class Score:
    def __init__(self, score: int, passed: bool):
        self.score = score
        self.passed = passed
    
    def to_dict(self):
        return {
            "score": self.score,
            "passed": self.passed,
        }
        
@dataclass(kw_only=True)
class TestResult:
    prompt_id: str
    questions: list[Question]
    score: Score