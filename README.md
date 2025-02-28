# LLM Prompt Evaluation Tool

This tool allows you to evaluate how well your LLM responses match an ideal answer by comparing generated questions and answers. You can use the tool either by calling the comparison functions manually in your code or by passing a CSV file containing test cases. The tool also supports visualizing your test scores on a chart, with results automatically grouped by different prompt IDs.

## Key Parameters

- **ideal_answer (required):**  
  The reference or "ideal" answer that your LLM response is compared against.  
  **Example:**  
  ```
  "Blockchain is like a digital ledger that everyone can see but no one can change."
  ```

- **llm_response (required):**  
  Your LLM's response.

- **optional_params (optional):**  
  A JSON-like dictionary that may include extra details for the test. It has the following structure:
  - `prompt`: A string with the prompt to be used (if you want to override the `prompt_id` prompt).
  - `context`: (Optional) Additional context for the prompt.
  - `prompt_id`: (Optional) A unique identifier for the prompt to be fetched online from Lamoom Service.  
  **Example:**  
  ```json
  {
    "prompt": "Explain blockchain to a beginner {username}",
    "context": {'username': 'James Bond'},
    "prompt_id": "beginner_blockchain"
  }
  ```

## Using the Tool

### 1. Manual Testing

You can manually call the `compare()` method by passing the required `ideal_answer` and `llm_response` and (optionally) `optional_params`. Each call will automatically accumulate the test results based on the provided (or default) `prompt_id` from `optional_params`.

**Example:**

```python
from lamoom_cicd import TestLLMResponsePipe
import time

ideal_answer = (
    "Blockchain is like a digital notebook that everyone can see, but no one can secretly change. "
    "Imagine a shared Google Doc where every change is recorded forever, and no one can edit past entries."
)

lamoom_pipe = TestLLMResponsePipe(openai_key=os.environ.get("OPENAI_KEY"))
# When llm_response is not passed, it defaults to None.
result = lamoom_pipe.compare(ideal_answer, "Your LLM response here")

# Print individual question details
for question in result.questions:
    print(question.to_dict())

# Print overall score details
print(result.score.to_dict())
```

### 2. Testing with CSV

You can also pass multiple test cases using a CSV file. The CSV file should contain the following columns:

- **ideal_answer:** (Required) The ideal answer text.
- **llm_response:** (Required) LLM response to compare with.
- **optional_params:** (Optional) A JSON string containing the optional parameters.  

Multiple rows can be included, and you can use different `prompt_id` values to test various prompts.

**Example CSV Content:**
IMPORTANT: take notice of double quotes when putting json in a csv file!

```csv
ideal_answer,llm_response, optional_params
"blockchain_prompt","Blockchain is a secure, immutable digital ledger.","Blockchain is like a shared Google Doc that records every change.","{""prompt_id"": ""google_doc_blockchain""}", 
```

**Usage Example:**

```python
csv_file_path = "test_data.csv"
lamoom_pipe = TestLLMResponsePipe(openai_key=os.environ.get("OPENAI_KEY"))
accumulated_results = lamoom_pipe.compare_from_csv("test_prompt", csv_file_path)
```

### 3. Visualizing Test Scores

After running tests (whether manually or using a CSV), the results are automatically accumulated by `prompt_id`. To see a visual chart of test scores, use the provided visualization function.

**Example:**

```python
lamoom_pipe.visualize_test_results()
```

This function will generate a line chart with the x-axis representing the test instance number (as integers) and the y-axis representing the score percentage. Each line on the chart corresponds to a different `prompt_id`.

## How does it work?


## Summary

- **ideal_answer**, **llm_response** are required parameters.
- **optional_params** are optional, with `optional_params` offering extra configuration (like a custom prompt and a unique `prompt_id` for tests).
- You can compare responses either manually or via CSV (which supports multiple test cases).
- The tool accumulates results for each `prompt_id` across multiple calls.
- Use the visualization function to see your test scores on an easy-to-read chart.

Enjoy using the tool to refine and evaluate your LLM prompts!