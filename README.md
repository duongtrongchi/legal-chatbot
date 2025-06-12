# Legal Chatbot

A machine learning project for training and experimenting with language models on legal datasets. Built with Python, [Unsloth](https://github.com/unslothai/unsloth), and Hugging Face Datasets.

## Features
- Easily configurable training and hyperparameters via YAML
- Supports large language models (LLMs) with efficient memory usage
- Ready for CI/CD with GitHub Actions

## Requirements
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended, but pip is supported)
- [Hugging Face account & token](https://huggingface.co/settings/tokens) (set as `HF_TOKEN` environment variable)

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd legal-chatbot
```

### 2. Install dependencies
#### With uv (recommended):
```bash
pip install uv
uv pip install --system --all-extras --requirement pyproject.toml
```
#### Or with pip:
```bash
pip install .
```

### 3. Set environment variables
Create a `.env` file or export your Hugging Face token:
```bash
export HF_TOKEN=your_hf_token_here
```

## Configuration
Edit `configs/training.yaml` to change model, hyperparameters, or training settings.

## Training
To start training:
```bash
make train
```
Or directly:
```bash
uv run src/training.py
```

## Testing
Add your tests to the `tests/` directory. To run tests:
```bash
pytest tests
```

## Continuous Integration (CI)
- GitHub Actions workflow runs on every push/PR to `main`
- Installs dependencies, lints code, and runs tests automatically

## Project Structure
```
legal-chatbot/
├── configs/           # YAML config files
├── outputs/           # Training outputs
├── src/               # Source code
├── tests/             # Tests
├── Makefile           # Common commands
├── pyproject.toml     # Python project metadata
└── README.md          # This file
```

## License
MIT (add your license here)
