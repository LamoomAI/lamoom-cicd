from lamoom import PipePrompt

agent = PipePrompt(id='compare_results')
agent.add("""
You will need to get from the text answers on the questions. If in text there is no answer please say: "ANSWER_NOT_PROVIDED".
For Each question there is an ideal answer. You need to compare the ideal answer and the answer you got from the text.
If they match logically then this question is a pass, otherwise no.
""", role='system')

agent.add("""
# TEXT
{user_prompt_response}
# QUESTIONS_AND_ANSWERS
{generated_test}
          
Please go through each question and answer it with the TEXT you have. And Compare the answer you got with the ideal answer from the generated test. Use the next JSON format for the answer:
# RESPONSE
```json
{
    "question": {
        "real_answer": "answer from the text",
        "ideal_answer": "ideal answer from the generated test",
        "does_match_with_ideal_answer": true/false
    },
    ...
}
```
""", role='user')
