Sure, here's the updated `README.md` content with the added future enhancement.

````markdown
# Graph Search & Visualization

A modern app for searching, extracting, and visualizing data using LLM-powered workflows.

---

## Features

* **Conditional graph generation** based on user query
* **Web search** and **data cleaning** capabilities
* **Automatic graph type selection** (bar, stacked bar, multi-bar, pie, line, scatter)
* **Streamlit web interface** for interactive use

---

## Quickstart

To get the application up and running quickly:

### Installation

1.  **Create and activate a virtual environment** using `uv` (requires Python 3.9+, recommended 3.12.10):
    ```sh
    uv venv -p 3.12.10
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

2.  **Install project dependencies**:
    ```sh
    uv pip install -r pyproject.toml
    ```

3.  **Set up .env file**:
    ```sh
    OPENAI_API_KEY = "sk-pro..
    ```

### Running the App

* **To run the main application**:
    ```sh
    streamlit run app.py
    ```

---

## Development

### Project Structure

* `app.py` — Streamlit app entry point
* `src/` — Contains core modules and workflow logic
* `tests/` — Houses Pytest-based tests

### Testing

Make sure `pytest` is installed (it's included if you installed with `--extra test`).

* **Run all tests**:
    ```sh
    pytest
    ```
* **Run tests with verbose output**:
    ```sh
    pytest -v
    ```
* **Run tests with coverage**:
    ```sh
    pytest --cov=src
    ```
* **Run a specific test file**:
    ```sh
    pytest tests/test_workflows.py
    ```
* **Run test on a specific testing function**:
    ```sh
    pytest tests/test_workflows.py::TestWorkflowIntegration::test_supported_graph_types -v
    ```

---

## Sample Queries

Here are some example queries you can try with the application:

* `Tell me the count of different nuclear warheads and nuclear reactors owned by different countries`

---

## Future Enhancements

1.  **Data Trimming**: Implement functionality to aggregate smaller, less significant data points into an 'Others' category for better graph readability.
2.  **Feedback Loops**: Add error correction feedback loops to mitigate LLM hallucinations.

---

````