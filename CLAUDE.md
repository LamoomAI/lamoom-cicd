# CLAUDE.md - LamoomAI CICD Project Guidelines

## Build & Test Commands
- Install dependencies: `poetry install`
- Run all tests: `make test`
- Run single test: `poetry run pytest tests/test_cicd.py::test_compare -v`
- Run specific file: `poetry run pytest tests/test_cicd.py -v`

## Code Style Guidelines
- **Formatting**: PEP 8 compliant code structure
- **Types**: Use explicit type annotations with dataclasses
- **Naming**:
  - Classes: PascalCase (e.g., `TestLLMResponsePipe`)
  - Functions/variables: snake_case (e.g., `compare_from_csv`)
- **Imports**: Group imports by standard library, third-party, then local
- **Error Handling**: Use custom exceptions (e.g., `GenerateFactsException`)
- **Documentation**: Use docstrings for functions and methods

## Project Structure
- Core functionality in `lamoom_cicd/`
- Tests in `tests/` directory with `test_` prefix
- Prompt templates in `lamoom_cicd/prompts/`
- Custom response classes for structured data handling