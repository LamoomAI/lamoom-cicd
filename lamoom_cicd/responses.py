from dataclasses import dataclass

class Question:
    def __init__(self, question: str, answer_by_llm: str, ideal_answer: str, match: bool):
        self.question = question
        self.answer_by_llm = answer_by_llm
        self.ideal_answer = ideal_answer
        self.match = match

    def to_dict(self):
        return {
            "question": self.question,
            "answer_by_llm": self.answer_by_llm,
            "ideal_answer": self.ideal_answer,
            "match": self.match
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