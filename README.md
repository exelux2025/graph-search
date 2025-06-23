# graph-search

# Install Venv
uv venv -p 3.12.10

# Activate venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r .\requirements.txt

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_workflows.py

# Run the app
streamlit run app.py