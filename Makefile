SHELL=/bin/bash
.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

integration_tests:
	poetry run pytest tests/integration_tests

test:
	poetry run pytest -v $(TEST_FILE)

tests: 
	poetry run pytest -v $(TEST_FILE)

test_watch:
	poetry run ptw --now . -- tests/unit_tests


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/experimental --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')

lint lint_diff:
	poetry run mypy $(PYTHON_FILES)
	poetry run black $(PYTHON_FILES) --check
	poetry run ruff .

format format_diff:
	poetry run black $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w


######################
# DOCUMENTATION
######################

clean: docs_clean api_docs_clean


docs_build:
	docs/.local_build.sh

docs_clean:
	rm -rf docs/_dist

docs_linkcheck:
	poetry run linkchecker docs/_dist/docs_skeleton/ --ignore-url node_modules

api_docs_build:
#	poetry run python docs/api_reference/create_api_rst.py
#	cd docs/api_reference && poetry run make html

api_docs_clean:
#	rm -f docs/api_reference/api_reference.rst
#	cd docs/api_reference && poetry run make clean


api_docs_linkcheck:
	poetry run linkchecker docs/api_reference/_build/html/index.html

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'clean                        - run docs_clean and api_docs_clean'
	@echo 'docs_build                   - build the documentation'
	@echo 'docs_clean                   - clean the documentation build artifacts'
	@echo 'docs_linkcheck               - run linkchecker on the documentation'
	@echo 'api_docs_build               - build the API Reference documentation'
	@echo 'api_docs_clean               - clean the API Reference documentation build artifacts'
	@echo 'api_docs_linkcheck           - run linkchecker on the API Reference documentation'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'


.PHONY: dist
dist:
	poetry build

# ---------------------------------------------------------------------------------------
# SNIPPET pour tester la publication d'une distribution
# sur test.pypi.org.
.PHONY: test-twine
## Publish distribution on test.pypi.org
test-twine: dist
ifeq ($(OFFLINE),True)
	@echo -e "$(red)Can not test-twine in offline mode$(normal)"
else
	@$(VALIDATE_VENV)
	rm -f dist/*.asc
	twine upload --sign --repository-url https://test.pypi.org/legacy/ \
		$(shell find dist -type f \( -name "*.whl" -or -name '*.gz' \) -and ! -iname "*dev*" )
endif

# ---------------------------------------------------------------------------------------
# SNIPPET pour publier la version sur pypi.org.
.PHONY: release
## Publish distribution on pypi.org
release: clean dist
ifeq ($(OFFLINE),True)
	@echo -e "$(red)Can not release in offline mode$(normal)"
else
	@$(VALIDATE_VENV)
	[[ $$( find dist -name "*.dev*" | wc -l ) == 0 ]] || \
		( echo -e "$(red)Add a tag version in GIT before release$(normal)" \
		; exit 1 )
	rm -f dist/*.asc
	echo "Enter Pypi password"
	twine upload  \
		$(shell find dist -type f \( -name "*.whl" -or -name '*.gz' \) -and ! -iname "*dev*" )

endif


sync:
	cp -rf ../langchain/libs/experimental/langchain_experimental/chains/qa_with_references/ \
		langchain_qa_with_references/chains/
	cp -f ../langchain/libs/experimental/langchain_experimental/chains/__init__.py \
		langchain_qa_with_references/chains/
	cp -rf ../langchain/libs/experimental/langchain_experimental/chains/qa_with_references_and_verbatims/ \
		langchain_qa_with_references/chains/
	cp -rf ../langchain/libs/experimental/tests/unit_tests/chains/ \
		tests/unit_tests/
	cp ../langchain/libs/experimental/docs/qa_with_reference*.ipynb .
	find . -type f \( -name '*.py' -or -name '*.ipynb' \) | xargs sed -i 's/langchain_experimental/langchain_qa_with_references/g'
	find . -type f -name '*.ipynb' | xargs sed -i 's/langchain\([_-]\)experimental/langchain\1qa_with_references/g'

reverse-sync:
	( cd langchain_qa_with_references/chains/ ; \
	 find . -type f -not -iname '__init__.py' -exec cp '{}' '../../../langchain/libs/experimental/langchain_experimental/chains/{}' ';' \
	)

	( cd tests/unit_tests/chains ; \
	 find . -type f -not -iname '__init__.py' -exec cp '{}' '../../../../langchain/libs/experimental/tests/unit_tests/chains/{}' ';' \
	)
	( cd tests/integration_tests/chains ; \
	 find . -type f -not -iname '__init__.py' -name '*_qa_with_*' -exec cp '{}' '../../../../langchain/libs/experimental/tests/integration_tests/chains/{}' ';' \
	)

	cp *.ipynb ../langchain/libs/experimental/docs/qa_with_reference*.ipynb
	find ../langchain/libs/experimental -type f \( -name '*.py' -or -name '*.ipynb' \) -exec  sed -i 's/langchain_qa_with_references/langchain_experimental/g' '{}' ';'
	find ../langchain/libs/experimental -type f \( -name '*.py' -or -name '*.ipynb' \) -exec  sed -i 's/langchain_experimental\.pydantic_v1/langchain\.pydantic_v1/g' '{}' ';'
	find ../langchain/libs/experimental/docs -type f -name '*.ipynb' -exec sed -i 's/langchain\([_-]\)qa_with_references/langchain\1experimental/g' '{}' ';'
