# Graph Search & Visualization

A modern app for searching, extracting, and visualizing data using LLM-powered workflows.

---

## Features

- **Conditional graph generation** based on user query  
- **Web search** and **data cleaning** capabilities  
- **Automatic graph type selection**  
  *(bar, stacked bar, multi-bar, pie, line, scatter)*  
- **Streamlit web interface** for interactive use  

---

## Quickstart

To get the application up and running:

### Installation

1. **Create and activate a virtual environment** using `uv` (Python 3.9+ recommended, tested on 3.12.10):

   ```bash
   uv venv -p 3.12.10

   # On Windows:
   .venv\Scripts\activate

   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install project dependencies**:

   ```bash
   uv pip install -r pyproject.toml
   ```

3. **Set up environment variables**:

   Create a `.env` file with the following:

   ```env
   OPENAI_API_KEY="sk-pro-..."
   ```

### Running the App

To launch the main application:

```bash
streamlit run app.py
```

---

## Project Structure

- `app.py` — Streamlit app entry point  
- `src/` — Core modules and workflow logic  
- `tests/` — Pytest-based unit and integration tests  

---

## Testing

Make sure `pytest` is installed (included if installed with `--extra test`).

Run all tests:

```bash
pytest
```

Verbose output:

```bash
pytest -v
```

With coverage:

```bash
pytest --cov=src
```

Run a specific file:

```bash
pytest tests/test_workflows.py
```

Run a specific test function:

```bash
pytest tests/test_workflows.py::TestWorkflowIntegration::test_supported_graph_types -v
```

---

## Sample Queries

Try queries like:

> Tell me the count of different nuclear warheads and nuclear reactors owned by different countries

---

## Future Enhancements

1. **Data Trimming**: Aggregate smaller, less significant data points into an `Others` category for improved chart readability.  
2. **Feedback Loops**: Add feedback loops to reduce LLM hallucination and improve output accuracy.  

---

## License

MIT License — open for use, contribution, and modification.
