# Legal Chatbot Documentation

Welcome to the documentation for the **Legal Chatbot** project. This project provides tools and scripts for training and experimenting with large language models (LLMs) on legal datasets, leveraging Unsloth and Hugging Face Datasets.

## Overview
- **Purpose:** Enable efficient training and experimentation with LLMs for legal text data.
- **Features:**
  - Configurable training and hyperparameters via YAML
  - Support for large models with memory optimizations
  - Integration with Hugging Face and Unsloth
  - Redis-based data storage utilities
  - CI/CD ready

## Project Structure
| Folder/File         | Description                        |
|--------------------|------------------------------------|
| `src/`             | Main source code                   |
| `configs/`         | YAML configuration files            |
| `outputs/`         | Training outputs                    |
| `tests/`           | Unit tests                          |
| `docs/`            | Project documentation (this folder) |

## Documentation Sections
- [Configuration & Settings](configuration.md)
- [Training Pipeline](training.md)
- [Data Processing](processing.md)
- [Utilities](utils.md)
- [Redis Integration](redis_handler.md)

---

For installation, setup, and usage, see the [README](../README.md).