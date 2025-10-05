VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
COVERAGE := $(PYTHON) -m coverage
TESTS_ARGS := -m unittest discover

all: venv coverage train predict

clean:
	find . -type d -name '__pycache__' -prune -execdir rm -rf {} +
	rm -rf logs models .coverage coverage.xml

coverage:
	$(COVERAGE) run $(TESTS_ARGS)
	$(COVERAGE) xml
	$(COVERAGE) report -m --skip-covered

predict:
	$(PYTHON) ft_linear_regression.py predict -m models/data.json

train:
	$(PYTHON) ft_linear_regression.py train -d datasets/data.csv --plot

test:
	$(PYTHON) $(TESTS_ARGS)

venv: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $<

.PHONY: all clean coverage predict train test venv
