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