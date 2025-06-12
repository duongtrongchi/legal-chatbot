TRAINING_SCRIPT=src/training.py

setup:
	pip install uv
	uv pip install --all-extras --requirement pyproject.toml

train:
	uv run $(TRAINING_SCRIPT)