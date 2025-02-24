from lamoom import PipePrompt

agent = PipePrompt(id='generate_facts')
agent.add("""
# IDEAL_ANSWER:
{ideal_answer}
Please come up with a list of facts that can be extracted from the IDEAL_ANSWER. 
generated_name - When? What? Why? How? Where? ... Generate a 3-7 word name containing the specific of the usecase based on your generated questions & answers. Do not include any personal information, like names... Think that you need to highlight specifics, more likely that prompt will have 100 tests.
""", role='system')

agent.add("""
First, Please highlights AS MUCH important statements AS YOU CAN from the IDEAL_ANSWER, based on which you can compare the generated content.
Secondly, write questions that will give facts as answers if you ask the ideal answer these questions.
Use the next json format for the answer:
```json
{
    "statements": [
        "statement_from_ideal_answer",
        ...
        "another_statement_from_ideal_answer"
    ],
    "questions": {
        "statement_from_ideal_answer": "question_to_answer_by_that_statement",
        "another_statement_from_ideal_answer": "question_to_answer_by_that_another_statement",
        ...,
        "qN": "statementN",
    }
    "name": "when_what_why_generated_name"
}
```
""")

