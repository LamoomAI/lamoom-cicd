import json
import logging
import os
from lamoom import Lamoom
from lamoom.response_parsers.response_parser import get_json_from_response
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Tuple
import random
import statistics

from lamoom_cicd.prompts.prompt_generate_facts import agent as generate_facts_agent
from lamoom_cicd.prompts.prompt_compare_results import agent as compare_results_agent

from lamoom_cicd.responses import Question, TestResult, Score, TestCase, AggregatedResult
from lamoom_cicd.exceptions import GenerateFactsException

from lamoom_cicd.utils import (
    parse_csv_file, 
    save_to_json_file, 
    load_from_json_file,
    load_objects_from_json,
    generate_case_id
)

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
    llm_model: str = "openai/o3-mini"
    
    accumulated_results: list[TestResult] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)
    aggregated_results: list[AggregatedResult] = field(default_factory=list)
    
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
                optional_params: dict = None,
                run_id: str = None,
                existing_questions: List[Dict[str, str]] = None) -> TestResult:
        """
        Compare LLM response against an ideal answer
        
        Args:
            ideal_answer: The ideal/expected answer
            llm_response: The LLM generated response to test
            optional_params: Optional parameters (prompt_id, etc.)
            run_id: Unique identifier for this test run
            existing_questions: Pre-generated test questions to use instead of generating new ones
            
        Returns:
            TestResult object containing the comparison results
        """
        lamoom = Lamoom(openai_key=self.openai_key, api_token=self.lamoom_token)
        
<<<<<<< Updated upstream
        # Generate test questions based on the ideal answer
        response = lamoom.call(generate_facts_agent.id, {"ideal_answer": ideal_answer}, "openai/o3-mini")

        result = get_json_from_response(response).parsed_content
        statements, questions = result.get("statements"), result.get("questions")
        generated_test = json.dumps(self.get_generated_test(statements, questions))
        user_prompt_response = llm_response
=======
>>>>>>> Stashed changes
        prompt_id = "user_prompt"
        if optional_params is not None:
            logger.info(optional_params)
            prompt_id = optional_params.get('prompt_id', "user_prompt")
            
        # Use existing questions or generate new ones
        if existing_questions:
            # Use pre-generated questions
            test_questions = {}
            for question_data in existing_questions:
                question = question_data.get("test_question")
                ideal_answer = question_data.get("ideal_answer")
                print(f'Question {question} - Ideal answer {ideal_answer}')

        else:
            # Generate test questions based on the ideal answer
            response = lamoom.call(generate_facts_agent.id, {"ideal_answer": ideal_answer}, self.llm_model)
            result = get_json_from_response(response).parsed_content
            statements, questions = result.get("statements"), result.get("questions")
            generated_test = self.get_generated_test(questions)
        
        # Compare results
        comparison_context = {
            "generated_test": json.dumps(generated_test),
            "user_prompt_response": llm_response,
        }

        comparison_response = lamoom.call(compare_results_agent.id, comparison_context, self.llm_model)
        test_results = get_json_from_response(comparison_response).parsed_content

        # Format results into Question objects
        questions_list = [
            Question(q, v["real_answer"], v["ideal_answer"], v["does_match_with_ideal_answer"])
            for q, v in test_results.items()
        ]
        
        score = self.calculate_score(test_results, self.threshold)
        
<<<<<<< Updated upstream
        test_result = TestResult(prompt_id=prompt_id, questions=questions_list, score=score)
=======
        test_result = TestResult(
            prompt_id=prompt_id,
            questions=questions_list,
            score=score,
            ideal_response=ideal_answer,
            llm_response=llm_response,
            optional_params=optional_params,
            run_id=run_id
        )
>>>>>>> Stashed changes
        self.accumulated_results.append(test_result)

        return test_result
    
    def save_test_case(self, test_result: TestResult, case_id: str = None) -> TestCase:
        """
        Save test questions from a test result as a reusable test case
        
        Args:
            test_result: The TestResult to save as a test case
            case_id: Optional case ID (generated if not provided)
            
        Returns:
            TestCase object
        """
        if not case_id:
            case_id = generate_case_id()
            
        # Extract questions and their ideal answers
        test_questions = []
        for question in test_result.questions:
            test_questions.append({
                "test_question": question.test_question,
                "ideal_answer": question.ideal_answer
            })
            
        # Create test case
        test_case = TestCase(
            case_id=case_id,
            prompt_id=test_result.prompt_id,
            ideal_response=test_result.ideal_response,
            test_questions=test_questions,
            context=test_result.optional_params or {}
        )
        
        self.test_cases.append(test_case)
        return test_case
    
    def save_test_cases_to_json(self, file_path: str) -> bool:
        """
        Save all test cases to a JSON file
        
        Args:
            file_path: Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        data = [case.to_dict() for case in self.test_cases]
        return save_to_json_file(file_path, data)
    
    def load_test_cases_from_json(self, file_path: str) -> List[TestCase]:
        """
        Load test cases from a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of TestCase objects
        """
        cases = load_objects_from_json(file_path, TestCase)
        self.test_cases = cases
        return cases
    
    def save_test_results_to_json(self, file_path: str) -> bool:
        """
        Save all test results to a JSON file
        
        Args:
            file_path: Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        data = [result.to_dict() for result in self.accumulated_results]
        return save_to_json_file(file_path, data)
    
    def load_test_results_from_json(self, file_path: str) -> List[TestResult]:
        """
        Load test results from a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of TestResult objects
        """
        results = load_objects_from_json(file_path, TestResult)
        self.accumulated_results = results
        return results
    
    def run_test_case(self, case_id: str, llm_response: str, 
                     run_id: str = None) -> TestResult:
        """
        Run a single test using a saved test case
        
        Args:
            case_id: ID of the test case to run
            llm_response: LLM response to test
            run_id: Optional unique run identifier
            
        Returns:
            TestResult with the test results
        """
        # Find the test case
        test_case = next((case for case in self.test_cases if case.case_id == case_id), None)
        if not test_case:
            raise ValueError(f"Test case with ID {case_id} not found")
            
        # Run the test using the pre-generated questions
        return self.compare(
            ideal_answer=test_case.ideal_response,
            llm_response=llm_response,
            optional_params=test_case.context,
            run_id=run_id,
            existing_questions=test_case.test_questions
        )
    
    def run_multiple_tests(self, case_id: str, llm_responses: List[str], 
                          runs: int = None) -> List[TestResult]:
        """
        Run multiple tests of the same test case with different LLM responses
        
        Args:
            case_id: ID of the test case to run
            llm_responses: List of LLM responses to test
            runs: Number of runs to perform (if not specified, uses length of llm_responses)
            
        Returns:
            List of TestResult objects
        """
        if runs is None:
            runs = len(llm_responses)
            
        # Make sure we have enough responses
        if len(llm_responses) < runs:
            raise ValueError(f"Not enough LLM responses provided. Need {runs}, got {len(llm_responses)}")
            
        results = []
        for i in range(runs):
            run_id = f"{case_id}-run-{i+1}"
            result = self.run_test_case(
                case_id=case_id,
                llm_response=llm_responses[i],
                run_id=run_id
            )
            results.append(result)
            
        return results
        
    def run_tests(self, 
                  llm_response_provider: Union[str, List[str], Dict[str, List[str]]], 
                  test_cases: Union[str, List[TestCase]] = None,
                  runs_per_case: int = 1,
                  save_results: bool = True,
                  results_file: str = None) -> Dict[str, Any]:
        """
        Comprehensive test runner that can handle various input forms for maximum flexibility.
        
        This method provides a streamlined, user-friendly interface for running tests with various
        configurations:
        
        1. Run tests from a test cases file with one or more LLM responses
        2. Run tests from already loaded test cases with one or more LLM responses
        3. Run tests with a mapping of case_ids to specific LLM responses
        
        Args:
            llm_response_provider: Can be one of:
                - String: Path to a JSON file containing LLM responses mapped to case_ids
                - List[str]: List of LLM responses to test with each test case
                - Dict[str, List[str]]: Mapping of case_ids to lists of LLM responses
            
            test_cases: Can be one of:
                - String: Path to a JSON file containing test cases
                - List[TestCase]: List of already loaded test cases
                - None: Use already loaded test cases in self.test_cases
                
            runs_per_case: Number of runs per test case (only used when llm_response_provider is a list)
            
            save_results: Whether to save results to a file
            
            results_file: Path to save results (if None, generates a timestamped filename)
            
        Returns:
            Dictionary with:
                - results: List of all TestResult objects
                - aggregated: List of AggregatedResult objects
                - summary: Dictionary with overall statistics
        """
        import time
        from datetime import datetime
        
        # Step 1: Prepare test cases
        if isinstance(test_cases, str):
            # Load test cases from file
            loaded_cases = self.load_test_cases_from_json(test_cases)
            working_test_cases = loaded_cases
        elif isinstance(test_cases, list) and all(isinstance(tc, TestCase) for tc in test_cases):
            # Use provided test cases
            working_test_cases = test_cases
        elif test_cases is None:
            # Use already loaded test cases
            working_test_cases = self.test_cases
        else:
            raise ValueError("test_cases must be a file path, list of TestCase objects, or None")
            
        if not working_test_cases:
            raise ValueError("No test cases found. Please provide test cases or load them first.")
            
        # Step 2: Prepare LLM responses
        case_to_responses = {}
        
        if isinstance(llm_response_provider, str):
            # Load responses from file
            with open(llm_response_provider, 'r') as f:
                case_to_responses = json.load(f)
                
            # Validate the loaded data
            if not isinstance(case_to_responses, dict):
                raise ValueError(f"LLM responses file must contain a dictionary mapping case_ids to lists of responses")
                
        elif isinstance(llm_response_provider, list):
            # Same responses for all test cases
            for case in working_test_cases:
                # If we don't have enough responses for the requested runs, repeat the last one
                if len(llm_response_provider) < runs_per_case:
                    extended_responses = llm_response_provider.copy()
                    while len(extended_responses) < runs_per_case:
                        extended_responses.append(llm_response_provider[-1])
                    case_to_responses[case.case_id] = extended_responses[:runs_per_case]
                else:
                    case_to_responses[case.case_id] = llm_response_provider[:runs_per_case]
                    
        elif isinstance(llm_response_provider, dict):
            # Specific responses for specific cases
            case_to_responses = llm_response_provider
            
            # Validate all case_ids exist
            for case_id in case_to_responses:
                if not any(case.case_id == case_id for case in working_test_cases):
                    logger.warning(f"Test case {case_id} not found in loaded test cases")
        else:
            raise ValueError("llm_response_provider must be a file path, list of responses, or dictionary mapping case_ids to responses")
            
        # Step 3: Run tests
        all_results = []
        all_aggregated = []
        
        print(f"Running tests for {len(working_test_cases)} test cases...")
        start_time = time.time()
        
        for case in working_test_cases:
            if case.case_id not in case_to_responses:
                logger.warning(f"No responses provided for test case {case.case_id}, skipping")
                continue
                
            responses = case_to_responses[case.case_id]
            if not responses:
                logger.warning(f"Empty responses list for test case {case.case_id}, skipping")
                continue
                
            print(f"Running {len(responses)} tests for case {case.case_id}...")
            
            # Run tests for this case
            results = self.run_multiple_tests(case.case_id, responses)
            all_results.extend(results)
            
            # Aggregate results
            if len(results) > 0:
                aggregated = self.aggregate_results(case.case_id, results)
                all_aggregated.append(aggregated)
                
                print(f"  Average score: {aggregated.avg_score:.2f}% (±{aggregated.std_deviation:.2f})")
                print(f"  Pass rate: {aggregated.pass_rate:.2f}%")
                
        elapsed_time = time.time() - start_time
        print(f"Completed all tests in {elapsed_time:.2f} seconds")
        
        # Step 4: Save results if requested
        if save_results:
            if results_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"test_results_{timestamp}.json"
                
            self.save_test_results_to_json(results_file)
            print(f"Saved {len(all_results)} test results to {results_file}")
            
        # Step 5: Create summary
        overall_scores = [result.score.score for result in all_results]
        pass_count = sum(1 for result in all_results if result.score.passed)
        
        summary = {
            "total_tests": len(all_results),
            "test_cases": len(working_test_cases),
            "overall_pass_rate": (pass_count / len(all_results) * 100) if all_results else 0,
            "avg_score": statistics.mean(overall_scores) if overall_scores else 0,
            "min_score": min(overall_scores) if overall_scores else 0,
            "max_score": max(overall_scores) if overall_scores else 0,
            "std_deviation": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            "threshold": self.threshold,
            "elapsed_time": elapsed_time
        }
        
        return {
            "results": all_results,
            "aggregated": all_aggregated,
            "summary": summary
        }
        
    def save_llm_responses(self, responses_dict: Dict[str, List[str]], file_path: str) -> bool:
        """
        Save a dictionary of case_ids to LLM responses for later testing
        
        Args:
            responses_dict: Dictionary mapping case_ids to lists of LLM responses
            file_path: Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        return save_to_json_file(file_path, responses_dict)
        
    def load_llm_responses(self, file_path: str) -> Dict[str, List[str]]:
        """
        Load a dictionary of case_ids to LLM responses for testing
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary mapping case_ids to lists of LLM responses
        """
        data = load_from_json_file(file_path)
        if not data:
            return {}
            
        return data
    
    def aggregate_results(self, case_id: str, results: List[TestResult] = None) -> AggregatedResult:
        """
        Aggregate results from multiple test runs
        
        Args:
            case_id: ID of the test case
            results: List of TestResult objects (if None, uses accumulated_results)
            
        Returns:
            AggregatedResult with statistics
        """
        if results is None:
            # Filter accumulated results by case_id
            results = [r for r in self.accumulated_results 
                      if r.run_id and r.run_id.startswith(f"{case_id}-run-")]
            
        if not results:
            raise ValueError(f"No results found for case ID {case_id}")
            
        aggregated = AggregatedResult.from_test_results(results, case_id)
        self.aggregated_results.append(aggregated)
        return aggregated
    
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
            results.append(test_result)
        
        return results
    
    def generate_tests(self, ideal_answer: str, optional_params: Dict[str, Any] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate test questions from an ideal answer without performing comparison
        
        Args:
            ideal_answer: The ideal answer text to generate questions from
            optional_params: Optional parameters dictionary which may contain:
                - prompt_id: Identifier for the prompt
                - context: Additional context information
                - prompt_data: The original prompt data
                
        Returns:
            Dictionary with generated questions and answers:
                {
                    "test_questions": [
                        {"test_question": str, "ideal_answer": str},
                        ...
                    ],
                    "prompt_id": str,
                    "context": Dict
                }
        """
        lamoom = Lamoom(openai_key=self.openai_key, api_token=self.lamoom_token)
        
        # Extract prompt_id from optional params
        prompt_id = "user_prompt"
        context = {}
        if optional_params is not None:
            prompt_id = optional_params.get('prompt_id', "user_prompt")
            context = optional_params.copy()
        
        # Generate test questions based on the ideal answer
        response = lamoom.call(generate_facts_agent.id, {"ideal_answer": ideal_answer}, self.llm_model)
        result = get_json_from_response(response).parsed_content
        statements, questions = result.get("statements"), result.get("questions")
        
        # Format into test questions format
        test_questions = []
        for statement, question in questions.items():
            test_questions.append({
                "test_question": question,
                "ideal_answer": statement
            })
            
        # Return structured result
        return {
            "test_questions": test_questions,
            "prompt_id": prompt_id,
            "context": context
        }
    
    def create_test_case(self, ideal_answer: str, 
                         optional_params: Dict[str, Any] = None, 
                         case_id: str = None) -> TestCase:
        """
        Create a test case directly from an ideal answer without running a comparison
        
        Args:
            ideal_answer: The ideal answer text to generate questions from
            optional_params: Optional parameters (prompt_id, context, etc.)
            case_id: Optional case ID (generated if not provided)
            
        Returns:
            TestCase object
        """
        if not case_id:
            case_id = generate_case_id()
            
        # Generate the test questions
        generated_data = self.generate_tests(ideal_answer, optional_params)
        
        # Create the test case
        test_case = TestCase(
            case_id=case_id,
            prompt_id=generated_data["prompt_id"],
            ideal_response=ideal_answer,
            test_questions=generated_data["test_questions"],
            context=generated_data["context"]
        )
        
        self.test_cases.append(test_case)
        return test_case
    
    def save_test_cases_from_csv(self, csv_file: str, output_file: str) -> List[TestCase]:
        """
        Generate and save test cases from a CSV file
        
        Args:
            csv_file: Path to CSV file with ideal answers
            output_file: Path to save JSON test cases
            
        Returns:
            List of generated TestCase objects
        """
        test_cases = parse_csv_file(csv_file)
        generated_cases = []
        
        for row in test_cases:
            ideal_answer = row.get("ideal_answer")
            optional_params = row.get("optional_params")
            
            # Create a test case directly without comparison
            test_case = self.create_test_case(ideal_answer, optional_params)
            generated_cases.append(test_case)
            
        # Save all test cases to JSON
        self.save_test_cases_to_json(output_file)
        return generated_cases
    
    def visualize_test_results(self, show_aggregated: bool = False):
        """
        Plots a line chart of accumulated scores grouped by prompt_id.
        
        Args:
            show_aggregated: If True, shows aggregated results with error bars
        """
        plt.figure(figsize=(12, 8))
        
        if show_aggregated and self.aggregated_results:
            # Plot aggregated results with error bars
            case_ids = []
            avg_scores = []
            std_devs = []
            colors = plt.cm.tab10(range(len(self.aggregated_results)))
            
            for i, result in enumerate(self.aggregated_results):
                case_ids.append(result.case_id)
                avg_scores.append(result.avg_score)
                std_devs.append(result.std_deviation)
                
                # Plot individual points for each run
                plt.scatter(
                    [result.case_id] * len(result.scores),
                    result.scores,
                    alpha=0.5,
                    color=colors[i % 10]
                )
                
            # Plot average scores with error bars
            plt.errorbar(
                case_ids,
                avg_scores,
                yerr=std_devs,
                fmt='o',
                capsize=5,
                elinewidth=2,
                markeredgewidth=2
            )
            
            plt.title(f"Aggregated Test Results (Passing score = {self.threshold}%)")
            plt.xlabel("Test Case ID")
            plt.ylabel("Score (%)")
            plt.grid(True)
            
        else:
            # Group scores by prompt_id (original visualization)
            groups = defaultdict(list)
            for item in self.accumulated_results:
                prompt_id = item.prompt_id
                score = item.score.score
                groups[prompt_id].append(score)

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
            
    def visualize_aggregated_results(self):
        """
        Plots a chart specifically for aggregated results with statistics
        """
        if not self.aggregated_results:
            logger.warning("No aggregated results to visualize")
            return
            
        plt.figure(figsize=(14, 10))
        
        # Create a bar chart for average scores
        case_ids = [result.case_id for result in self.aggregated_results]
        avg_scores = [result.avg_score for result in self.aggregated_results]
        std_devs = [result.std_deviation for result in self.aggregated_results]
        pass_rates = [result.pass_rate for result in self.aggregated_results]
        
        # Plot average scores with error bars
        ax1 = plt.subplot(2, 1, 1)
        bars = ax1.bar(
            case_ids,
            avg_scores,
            yerr=std_devs,
            capsize=5,
            alpha=0.7
        )
        
        # Add threshold line
        ax1.axhline(y=self.threshold, color='r', linestyle='--', label=f"Passing threshold ({self.threshold}%)")
        
        ax1.set_title("Average Scores with Standard Deviation")
        ax1.set_xlabel("Test Case ID")
        ax1.set_ylabel("Average Score (%)")
        ax1.set_ylim(0, 100)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot pass rates
        ax2 = plt.subplot(2, 1, 2)
        ax2.bar(
            case_ids,
            pass_rates,
            alpha=0.7,
            color='green'
        )
        
        ax2.set_title("Pass Rate Across Multiple Runs")
        ax2.set_xlabel("Test Case ID")
        ax2.set_ylabel("Pass Rate (%)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()