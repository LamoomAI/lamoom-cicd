from lamoom import PipePrompt

agent = PipePrompt(id='compare_results')
agent.add("""
You need to find answers on question in TEXT_TO_ASK_QUESTIONS. Ask Question from QUESTIONS_AND_ANSWERS. QUESTIONS_AND_ANSWERS has ideal answer and questions;
For Each question there is an ideal answer and real_answer. You need to compare the ideal_answer and the real_answer.
If they match logically then this question matches, otherwise no.
If real_answer doesnt match ideal answer, say: "ANSWER_NOT_PROVIDED".
""", role='system')

agent.add("""
# QUESTIONS_AND_ANSWERS
{generated_test}

# TEXT_TO_ASK_QUESTIONS
{user_prompt_response}

First, go through each question and get real_answer on it with from TEXT_TO_ASK_QUESTIONS. Compare the answer you got from QUESTIONS_AND_ANSWERS. Use the next JSON format for the answer:
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
