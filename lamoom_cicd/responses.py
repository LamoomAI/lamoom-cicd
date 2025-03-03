from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import uuid
import statistics

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
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            test_question=data["test_question"],
            llm_answer=data["llm_answer"],
            ideal_answer=data["ideal_answer"],
            does_match_ideal_answer=data["does_match_ideal_answer"]
        )
        
class Score:
    def __init__(self, score: int, passed: bool):
        self.score = score
        self.passed = passed
    
    def to_dict(self):
        return {
            "score": self.score,
            "passed": self.passed,
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Score':
        return cls(
            score=data["score"],
            passed=data["passed"]
        )
        
@dataclass(kw_only=True)
class TestResult:
    prompt_id: str
    questions: list[Question]
<<<<<<< Updated upstream
    score: Score
=======
    score: Score
    ideal_response: str
    llm_response: str
    optional_params: dict = None
    run_id: str = None

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "questions": [question.to_dict() for question in self.questions],
            "score": self.score.to_dict(),
            "ideal_response": self.ideal_response,
            "llm_response": self.llm_response,
            "optional_params": self.optional_params,
            "run_id": self.run_id
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        return cls(
            prompt_id=data["prompt_id"],
            questions=[Question.from_dict(q) for q in data["questions"]],
            score=Score.from_dict(data["score"]),
            ideal_response=data["ideal_response"],
            llm_response=data["llm_response"],
            optional_params=data.get("optional_params"),
            run_id=data.get("run_id")
        )

@dataclass
class TestCase:
    """Stores reusable test cases with questions and answers"""
    case_id: str
    prompt_id: str
    ideal_response: str
    test_questions: List[Dict[str, str]]  # List of {"test_question": str, "ideal_answer": str}
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt_id": self.prompt_id,
            "ideal_response": self.ideal_response,
            "test_questions": self.test_questions,
            "context": self.context
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        return cls(
            case_id=data["case_id"],
            prompt_id=data["prompt_id"],
            ideal_response=data["ideal_response"],
            test_questions=data["test_questions"],
            context=data.get("context", {})
        )

@dataclass
class AggregatedResult:
    """Statistics for multiple runs of the same test case"""
    case_id: str
    prompt_id: str
    runs: int
    avg_score: float
    min_score: float
    max_score: float
    std_deviation: float
    pass_rate: float  # Percentage of runs that passed
    scores: List[int]
    
    @classmethod
    def from_test_results(cls, results: List[TestResult], case_id: str) -> 'AggregatedResult':
        if not results:
            raise ValueError("Cannot aggregate empty results list")
            
        scores = [result.score.score for result in results]
        passed = sum(1 for result in results if result.score.passed)
        
        return cls(
            case_id=case_id,
            prompt_id=results[0].prompt_id,
            runs=len(results),
            avg_score=statistics.mean(scores),
            min_score=min(scores),
            max_score=max(scores),
            std_deviation=statistics.stdev(scores) if len(scores) > 1 else 0,
            pass_rate=(passed / len(results) * 100),
            scores=scores
        )
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt_id": self.prompt_id,
            "runs": self.runs,
            "avg_score": self.avg_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "std_deviation": self.std_deviation,
            "pass_rate": self.pass_rate,
            "scores": self.scores
        }
>>>>>>> Stashed changes
