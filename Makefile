.PHONY: install run test clean

NOTEBOOK = notebooks/001_eda.ipynb

install:
	pip install -r requirements.txt

run:
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
