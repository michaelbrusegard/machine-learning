# Machine Learning AIS Vessel

## Getting Started

### Setup a Virtual Environment

Ensure you have python 3.12 installed on your system. You can check by running:

```bash
python3 --version
```

To create a venv named machine-learning run:

```bash
python3 -m venv machine-learning
```

To activate the venv:

```bash
source machine-learning/bin/activate
```

#### Pyenv Alternative Setup

You can also use a tool like [pyenv](https://github.com/pyenv/pyen) with the [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) to handle python versions and venvs for you. It will activate the venv automaticaly for you when you enter the repository.

```bash
pyenv install 3.12

pyenv virtualenv 3.12 machine-learning

pyenv activate machine-learning
```

### Install dependencies

After activating the venv install dependencies:

```bash
pip install -r requirements.txt
```

### Pre commit hooks

Optionally you can install pre-commit hooks that automatically will format and check linting with ruff when you commit:

```bash
pre-commit install
```

You can also install [Ruff](https://docs.astral.sh/ruff/editors/setup/) for your code editor to automatically format and check linting.
