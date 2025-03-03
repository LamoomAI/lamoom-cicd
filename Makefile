PROJECT_FOLDER = 'lamoom_cicd'

test:
	poetry run pytest --cache-clear -vv tests \
		--cov=${PROJECT_FOLDER} \
		--cov-config=.coveragerc \
		--cov-fail-under=90 \
		--cov-report term-missing

publish-release:
	poetry config pypi-token.pypi "$(PYPI_API_KEY)"
	poetry version patch
	poetry build
	poetry publish


clean: clean-build clean-pyc clean-test

clean-build:
		rm -fr build/
		rm -fr dist/
		rm -fr .eggs/
		find . -name '*.egg-info' -exec rm -fr {} +
		find . -name '*.egg' -exec rm -f {} +

clean-pyc:
		find . -name '*.pyc' -exec rm -f {} +
		find . -name '*.pyo' -exec rm -f {} +
		find . -name '*~' -exec rm -f {} +
		find . -name '__pycache__' -exec rm -fr {} +

clean-test:
		rm -f .coverage
		rm -fr htmlcov/
		rm -rf .pytest_cache

