import json
import logging
from lamoom import Lamoom
from lamoom.response_parsers.response_parser import get_json_from_response
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt

from lamoom_cicd.prompts.prompt_generate_facts import agent as generate_facts_agent
from lamoom_cicd.prompts.prompt_compare_results import agent as compare_results_agent

from lamoom_cicd.responses import Question, TestResult, Score
from lamoom_cicd.exceptions import GenerateFactsException

from lamoom_cicd.utils import parse_csv_file

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TestLLMResponsePipe:
    lamoom_token: str = None
    openai_key: str = None
    azure_keys: dict = None
    gemini_key: str = None
    claude_key: str = None
    nebius_key: str = None
    
    threshold: int = 70
    
    accumulated_results: list[TestResult] = field(default_factory=list)
    
    def get_generated_test(self, statements: list, questions: dict):
        generated_test = {}
        for statement, question in questions.items():
            generated_test[question] = {
                'answer': statement,
                'required_to_pass': True
            }
        if len(statements) != len(questions):
            logger.error(f"Statements and questions are not equal in length. Statements: {len(statements)}: {statements}, Questions: {len(questions)}")
        if len(generated_test.items()) == 0:
            raise GenerateFactsException("No questions were generated.")
        
        return generated_test
    
    def calculate_score(self, test_results: dict, threshold: int) -> Score:
        pass_count = 0
        question_numb = len(test_results.items()) or 1
        for _, values in test_results.items():
            if values['does_match_with_ideal_answer']:
                pass_count += 1
            
        score = round(pass_count / question_numb * 100)
        passed = True if score >= threshold else False
        
        return Score(score, passed)
    
    def compare(self, ideal_answer: str, 
                llm_response: str, 
                optional_params: dict = None) -> TestResult:
        lamoom = Lamoom(openai_key=self.openai_key, api_token=self.lamoom_token)
        
        # Generate test questions based on the ideal answer
        response = lamoom.call(generate_facts_agent.id, {"ideal_answer": ideal_answer}, "openai/o3-mini")

        result = get_json_from_response(response).parsed_content
        statements, questions = result.get("statements"), result.get("questions")
        generated_test = json.dumps(self.get_generated_test(statements, questions))
        user_prompt_response = llm_response
        prompt_id = "user_prompt"
        if optional_params is not None:
            logger.info(optional_params)
            prompt_id = optional_params.get('prompt_id', "user_prompt")
            # TODO: Service CI/CD Logic

        # Compare results
        comparison_context = {
            "generated_test": generated_test,
            "user_prompt_response": user_prompt_response,
        }

        comparison_response = lamoom.call(compare_results_agent.id, comparison_context, "openai/o3-mini")
        test_results = get_json_from_response(comparison_response).parsed_content

        # Format results into Question objects
        questions_list = [
            Question(q, v["real_answer"], v["ideal_answer"], v["does_match_with_ideal_answer"])
            for q, v in test_results.items()
        ]
        
        score = self.calculate_score(test_results, self.threshold)
        
        test_result = TestResult(prompt_id=prompt_id, questions=questions_list, score=score)
        self.accumulated_results.append(test_result)

        return test_result
    
    
    def compare_from_csv(self, csv_file: str) -> list[TestResult]:
        """
        Reads a CSV file and runs compare() for each row.
        Expected CSV columns: ideal_answer, llm_response, optional_params
        (optional_params should be a valid JSON string if provided)
        Returns a list of test results.
        """
        test_cases = parse_csv_file(csv_file)
        results = []
        logger.info(f"CASES: {test_cases}")
        for row in test_cases:
            ideal_answer = row.get("ideal_answer")
            llm_response = row.get("llm_response")
            optional_params = row.get("optional_params")
            test_result = self.compare(ideal_answer, llm_response, optional_params)
            self.accumulated_results.append(test_result)
            results.append(test_result)
        
        return results
    
    def visualize_test_results(self):
        """
        Plots a line chart of accumulated scores grouped by prompt_id.
        """
        # Group scores by prompt_id.
        groups = defaultdict(list)
        for item in self.accumulated_results:
            prompt_id = item.prompt_id
            score = item.score.score
            groups[prompt_id].append(score)

        plt.figure(figsize=(10, 6))
        max_length = 0
        for prompt_id, scores in groups.items():
            x_values = list(range(1, len(scores) + 1))
            plt.plot(x_values, scores, marker='o', linestyle='-', label=f"Prompt: {prompt_id}")
            max_length = max(max_length, len(scores))

        plt.title(f"LLM Test Scores per Prompt (Passing score = {self.threshold}%)")
        plt.xlabel("Test Instance")
        plt.xticks(range(1, max_length + 1))
        plt.ylabel("Score (%)")
        plt.legend()
        plt.grid(True)
        plt.show()