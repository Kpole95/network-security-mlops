# Contributing to Phishing URL Detection — MLOps Pipeline

Thank you for your interest in contributing. This document covers setup,
code style, how to run tests, and how to submit changes.

---

## Table of contents

- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Development workflow](#development-workflow)
- [Code style](#code-style)
- [Running tests](#running-tests)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting bugs](#reporting-bugs)
- [Code of conduct](#code-of-conduct)

---

## Getting started

**Prerequisites:** Python 3.10+, MongoDB, and AWS credentials (for S3 artifact
sync and ECR deployment).

```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/YOUR_USERNAME/phishing-url-detector-mlops.git
cd phishing-url-detector-mlops

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your MongoDB URI and AWS credentials

# 5. Run the test suite to confirm everything works
pytest tests/ -v
```

---

## Project structure

```
src/
├── components/
│   ├── data_ingestion.py       # pulls data from MongoDB, splits train/test
│   ├── data_validation.py      # schema validation and drift detection (KS tests)
│   ├── data_transformation.py  # KNNImputer, ternary encoding, sklearn Pipeline
│   ├── model_trainer.py        # GridSearchCV over 5 classifiers, selects best
│   └── model_pusher.py         # syncs trained model artifact to AWS S3
├── pipeline/
│   ├── training_pipeline.py    # orchestrates all 5 components end to end
│   └── prediction_pipeline.py  # loads model from S3, runs inference
├── entity/
│   ├── config_entity.py        # typed config dataclasses per component
│   └── artifact_entity.py      # typed artifact contracts between components
├── exception.py                # structured exception handling
├── logger.py                   # centralized logging setup
└── utils/
    └── main_utils.py           # shared utilities (S3 read/write, yaml loading)

tests/
└── test_mongodb.py             # MongoDB connection and data retrieval tests

app.py                          # FastAPI inference service (POST /predict)
Dockerfile                      # containerized inference service
requirements.txt                # pinned dependencies
```

**Key design principle:** each component receives a typed config entity and
returns a typed artifact entity. This makes the pipeline modular — any
component can be replaced or tested in isolation without touching the others.

---

## Development workflow

```bash
# Run the full training pipeline
python src/pipeline/training_pipeline.py

# Start the FastAPI inference service locally
python app.py
# Then POST to http://localhost:8080/predict

# Run linting
ruff check .

# Run tests
pytest tests/ -v
```

---

## Code style

This project uses **ruff** for linting. Run before committing:

```bash
ruff check .
```

Key conventions:

- **Typed entities for all data contracts.** Every component input is a
  config entity, every output is an artifact entity. Do not pass raw dicts
  between pipeline stages.
- **Structured exception handling.** Use the custom `NetworkSecurityException`
  class rather than bare `raise`. Include the original exception and system
  info for traceability.
- **Centralized logging.** Use the logger from `src/logger.py`. Do not use
  `print()` in pipeline code.
- **No hardcoded credentials.** All secrets (MongoDB URI, AWS keys) must come
  from environment variables or `.env`. Never commit them.
- **One responsibility per component.** Each of the five pipeline components
  does exactly one thing. If you are adding logic that spans two components,
  reconsider the separation.

---

## Running tests

```bash
pytest tests/ -v
```

Tests require a running MongoDB instance with the phishing dataset loaded.
Set `MONGODB_URL_KEY` in your `.env` before running.

When adding a new component, add at least one test that verifies the component's
output artifact has the expected shape and content.

---

## Submitting a pull request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and confirm both checks pass:
   ```bash
   ruff check .
   pytest tests/ -v
   ```
3. Open a pull request against `main`. In the description:
   - Explain what changed and why
   - Link any related issues
   - If you changed the training pipeline, note the effect on model metrics

---

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include
reproduction steps, expected behavior, actual behavior, and your environment.

---

## Code of conduct

This project follows the
[Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Be respectful and constructive. Report unacceptable behavior to
krishnapole90@outlook.com.
