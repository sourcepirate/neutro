.PHONY: install test docs-check docs docs-serve examples hello

install:
	python -m pip install -e ".[test]"

test:
	pytest -q

docs-check:
	python scripts/check_docs.py

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

hello:
	python examples/00_hello_neutro.py

examples: hello
	@echo "Next: python examples/mnist_cnn.py"
