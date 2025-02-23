from dataclasses import dataclass

class Question:
    def __init__(self, question: str, answer: str, ideal_answer: str, comparison: bool):
        self.question = question
        self.answer = answer
        self.ideal_answer = ideal_answer
        self.comparison = comparison

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "ideal_answer": self.ideal_answer,
            "comparison": self.comparison
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