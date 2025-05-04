# Contributing to the MLOps Pipeline

Thank you for considering contributing to our MLOps pipeline project! This document outlines the process for contributing to this repository.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Create a new branch for your contribution
4. Make your changes
5. Push your changes to your fork
6. Submit a pull request

## Development Environment

To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb8-mm-mlops-pipeline.git
cd pb8-mm-mlops-pipeline

# Create and activate a virtual environment
conda create -n mlops-pipeline python=3.9
conda activate mlops-pipeline

# Install dependencies
pip install -r MLOps_Enhancements/requirements-enhancements.txt
pip install -e .

# Install development dependencies
pip install pytest flake8 black
```

## Code Style

We follow PEP 8 and use Black for code formatting:

```bash
# Format your code
black .

# Check for style issues
flake8 .
```

## Testing

Please write tests for any new features or bug fixes. Run the tests using pytest:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=MLOps_Enhancements
```

## Pull Request Process

1. Ensure your code passes all tests and style checks
2. Update the documentation if necessary
3. Make sure your commits have clear, descriptive messages
4. Create a pull request with a detailed description of your changes
5. Wait for a maintainer to review your PR
6. Address any feedback from the review

## Adding New Features

If you're adding a new feature:

1. First, open an issue describing the feature
2. Discuss the implementation approach with maintainers
3. Implement the feature with appropriate tests
4. Update the documentation to reflect the new feature

## Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Log files or error messages

## Code of Conduct

Please be respectful and inclusive in all interactions. We aim to foster an open and welcoming community.

## Questions?

If you have any questions, feel free to open an issue or contact the maintainers.

Thank you for your contributions! 