# LLM Prompt Evaluation Tool

This tool allows you to evaluate how well your LLM responses match an ideal answer by comparing generated questions and answers. You can use the tool either by calling the comparison functions manually in your code or by passing a CSV file containing test cases. The tool also supports visualizing your test scores on a chart, with results automatically grouped by different prompt IDs.

## Key Parameters

- **ideal_answer (required):**  
  The reference or "ideal" answer that your LLM response is compared against.  
  **Example:**  
  ```
  "Blockchain is like a digital ledger that everyone can see but no one can change."
  ```

- **llm_response (optional):**  
  Your LLM's response. If prompt is provided in `optional_params`, `llm_response` can be left as `None`.

- **optional_params (optional):**  
  A JSON-like dictionary that may include extra details for the test. It has the following structure:
  - `prompt`: A string with the prompt to be used (if you want to override the `prompt_id` prompt).
  - `context`: (Optional) Additional context for the prompt.
  - `prompt_id`: (Optional) A unique identifier for the prompt to be fetched online from Lamoom Service.  
  **Example:**  
  ```json
  {
    "prompt": "Explain blockchain to a beginner.",
    "context": {},
    "prompt_id": "beginner_blockchain"
  }
  ```

## Using the Tool

### 1. Manual Testing

You can manually call the `compare()` method by passing the required `ideal_answer` and (optionally) `llm_response` and `optional_params`. Each call will automatically accumulate the test results based on the provided (or default) `prompt_id`.

**Example:**

```python
ideal_answer = (
    "Blockchain is like a digital notebook that everyone can see, but no one can secretly change. "
    "Imagine a shared Google Doc where every change is recorded forever, and no one can edit past entries."
)
optional_params = {
    "prompt": "Explain the concept of blockchain to someone with no technical background."
}

test_response = TestLLMResponse(openai_key=os.environ.get("OPENAI_KEY"))
# When llm_response is not passed, it defaults to None.
result = test_response.compare(ideal_answer, optional_params=optional_params)

# Print individual question details
for question in result.questions:
    print(question.to_dict())

# Print overall score details
print(result.score.to_dict())
```

### 2. Testing with CSV

You can also pass multiple test cases using a CSV file. The CSV file should contain the following columns:

- **ideal_answer:** (Required) The ideal answer text.
- **optional_params:** (Optional) A JSON string containing the optional parameters.  
- **llm_response:** (Optional) If not provided, this value is set to `None`.

Multiple rows can be included, and you can use different `prompt_id` values to test various prompts.

**Example CSV Content:**

```csv
ideal_answer,optional_params,llm_response
"Blockchain is a secure, immutable digital ledger.","{\"prompt\": \"Explain blockchain simply.\", \"prompt_id\": \"simple_blockchain\"}", 
"Blockchain is like a shared Google Doc that records every change.","{\"prompt\": \"Describe blockchain in simple terms.\", \"prompt_id\": \"google_doc_blockchain\"}", 
```

**Usage Example:**

```python
csv_file_path = "test_data.csv"
test_response = TestLLMResponse(openai_key=os.environ.get("OPENAI_KEY"))
accumulated_results = test_response.compare_from_csv(csv_file_path)
```

### 3. Visualizing Test Scores

After running tests (whether manually or using a CSV), the results are automatically accumulated by `prompt_id`. To see a visual chart of test scores, use the provided visualization function.

**Example:**

```python
visualize_test_results(accumulated_results)
```

This function will generate a line chart with the x-axis representing the test instance number (as integers) and the y-axis representing the score percentage. Each line on the chart corresponds to a different `prompt_id`.

## Summary

- **ideal_answer** is the only required parameter.
- **llm_response** and **optional_params** are optional, with `optional_params` offering extra configuration (like a custom prompt and a unique `prompt_id`).
- You can compare responses either manually or via CSV (which supports multiple test cases).
- The tool accumulates results for each `prompt_id` across multiple calls.
- Use the visualization function to see your test scores on an easy-to-read chart.

Enjoy using the tool to refine and evaluate your LLM prompts!