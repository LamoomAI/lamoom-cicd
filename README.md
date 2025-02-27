# LLM Prompt Evaluation Tool

This tool allows you to evaluate how well your LLM response matches your ideal answer by comparing generated questions (from ideal answer) and answers (from llm response). The tool also supports visualizing your test scores on a chart, with results automatically grouped by different prompt IDs.

## Key Parameters

- **ideal_answer: str (required):**  
  The ideal answer for your prompt.

- **llm_response: str (required):**  
  Your LLM's response.

- **optional_params: dict (optional):**  
  A JSON-like dictionary that may include extra details for the test. For now you can pass `prompt_id` key to link test results to this `prompt_id` 
  **Example:**  
  ```json
  {
    "prompt_id": "blockchain_prompt"
  }
  ```

## Using the Tool


### 1. Pass ideal_answer, llm_response
User calls the `compare()` method by passing the `ideal_answer: str` and `llm_response: str` and (optionally) `optional_params: dict` parameters. `compare()` method returns an object with test results.

**Example:**

```python
from lamoom_cicd import TestLLMResponsePipe

# Users must provide these values (it doesn't matter how they obtain them)
ideal_answer = "Your ideal answer here"  # Replace with the expected response
llm_response = "Your LLM-generated response here"  # Replace with the model's actual response

# Optional: If you have a prompt_id, you can link results to it
optional_params = {
    "prompt_id": "your_prompt_id"  # Replace with an actual prompt identifier if needed
}

lamoom_pipe = TestLLMResponsePipe(openai_key=os.environ.get("OPENAI_KEY"))

# Compare the ideal answer with the LLM's response
result = lamoom_pipe.compare(ideal_answer, llm_response, optional_params=optional_params)

# Print test questions details
for question in result.questions:
    print(question.to_dict())

# Print final test score
print(result.score.to_dict())
```

### 2. Pass a CSV file
You can also pass multiple test cases using a CSV file. The CSV file should contain the following columns:

- **ideal_answer:** (Required) The ideal answer.
- **llm_response:** (Required) Your LLM response.
- **optional_params:** (Optional) A JSON string containing the optional parameters.  

If multiple rows are added then multiple test will run.

**Example CSV Content:**
IMPORTANT: take notice of double quotes when putting json in a csv file!

```csv
ideal_answer,llm_response, optional_params
"Blockchain is a secure, immutable digital ledger.","Blockchain is like a shared Google Doc that records every change.","{""prompt_id"": ""blockchain_prompt""}", 
```

**Usage Example:**

```python
csv_file_path = "your_file.csv"
lamoom_pipe = TestLLMResponsePipe(openai_key="your_key")
accumulated_results: list = lamoom_pipe.compare_from_csv(csv_file_path)
```

### 3. Visualizing Test Scores

After running tests, the results are automatically accumulated by `prompt_id`. To see a visual chart of test scores, use the provided visualization function.

**Example:**

```python
lamoom_pipe.visualize_test_results()
```

This function will generate a line chart with the x-axis representing the test instance number (as integers) and the y-axis representing the score percentage. Each line on the chart corresponds to a different `prompt_id`.

Enjoy using the tool to refine and evaluate your LLM prompts!