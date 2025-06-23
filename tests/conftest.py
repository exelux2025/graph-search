"""
Shared test fixtures and configuration for pytest.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import SystemMessage


@pytest.fixture
def sample_state():
    """Sample state for testing nodes."""
    return {
        "messages": [SystemMessage(content="You are a helpful assistant.")],
        "response": "",
        "search_results": "",
        "user_query": "Test query",
        "selected_graph_type": "",
        "formatted_data": "",
        "graph_object": None,
        "can_generate_graph": ""
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = Mock()
    mock_response.content = '{"can_generate_graph": "Yes", "reasoning": "Test reasoning"}'
    return mock_response


@pytest.fixture
def sample_json_data():
    """Sample JSON data for testing graph parsing."""
    return {
        "Country": {"values": ["USA", "China", "India"]},
        "Population": {"values": [331, 1441, 1380]}
    } 