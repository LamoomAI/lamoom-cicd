from lamoom import PipePrompt

agent = PipePrompt(id='lamoom_cicd__generate_facts')
agent.add("""
You're generating statements for the provided text below.
""", role='system')
agent.add("""
# TEXT_TO_GENERATE_STATEMENTS_TO:
{ideal_answer}
""", role='system')

agent.add("""
First, write out all the important statements from TEXT_TO_GENERATE_STATEMNTS_TO.
Second, ask questions to each generated statement so that it produces a statement as an answer to the question.
Third, give a name to the generated facts, such as the name of the test, like "when_what_why_how_who_what_generated_name".
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
    },
    "name": "when_what_why_generated_name"
}
```
""")

