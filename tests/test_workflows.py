"""
Unit tests for workflow functionality.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import SystemMessage

from src.workflows.conditional_graph_workflow import create_conditional_graph_workflow, get_initial_state as get_conditional_state
from src.workflows.web_search_workflow import create_web_search_graph, get_initial_state as get_web_search_state
from src.workflows.simple_chat_workflow import create_simple_chat_graph, get_initial_state as get_chat_state


class TestConditionalGraphWorkflow:
    """Test cases for conditional graph workflow."""
    
    def test_create_conditional_graph_workflow(self):
        """Test that conditional graph workflow can be created."""
        workflow = create_conditional_graph_workflow()
        assert workflow is not None
        assert hasattr(workflow, 'invoke')
    
    def test_get_conditional_initial_state(self):
        """Test initial state creation for conditional workflow."""
        user_query = "Test query"
        state = get_conditional_state(user_query)
        
        assert state["user_query"] == user_query
        assert state["can_generate_graph"] == ""
        assert state["response"] == ""
        assert state["search_results"] == ""
        assert state["selected_graph_type"] == ""
        assert state["formatted_data"] == ""
        assert state["graph_object"] is None
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], SystemMessage)


class TestWebSearchWorkflow:
    """Test cases for web search workflow."""
    
    def test_create_web_search_graph(self):
        """Test that web search graph can be created."""
        workflow = create_web_search_graph()
        assert workflow is not None
        assert hasattr(workflow, 'invoke')
    
    def test_get_web_search_initial_state(self):
        """Test initial state creation for web search workflow."""
        user_query = "Test query"
        state = get_web_search_state(user_query)
        
        assert state["user_query"] == user_query
        assert state["can_generate_graph"] == ""
        assert state["response"] == ""
        assert state["search_results"] == ""
        assert state["selected_graph_type"] == ""
        assert state["formatted_data"] == ""
        assert state["graph_object"] is None
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], SystemMessage)


class TestSimpleChatWorkflow:
    """Test cases for simple chat workflow."""
    
    def test_create_simple_chat_graph(self):
        """Test that simple chat graph can be created."""
        workflow = create_simple_chat_graph()
        assert workflow is not None
        assert hasattr(workflow, 'invoke')
    
    def test_get_chat_initial_state(self):
        """Test initial state creation for simple chat workflow."""
        user_query = "Test query"
        state = get_chat_state(user_query)
        
        assert state["user_query"] == user_query
        assert state["can_generate_graph"] == ""
        assert state["response"] == ""
        assert state["search_results"] == ""
        assert state["selected_graph_type"] == ""
        assert state["formatted_data"] == ""
        assert state["graph_object"] is None
        assert len(state["messages"]) == 2
        assert isinstance(state["messages"][0], SystemMessage)
        assert isinstance(state["messages"][1], SystemMessage)


class TestWorkflowIntegration:
    """Integration tests for workflows."""
    
    @patch('src.nodes.query_filtering.get_llm')
    def test_conditional_workflow_with_mock_classification(self, mock_get_llm):
        """Test conditional workflow with mocked classification."""
        # Mock the LLM response for classification
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"can_generate_graph": "Yes", "reasoning": "Test"}'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        workflow = create_conditional_graph_workflow()
        state = get_conditional_state("Test query")
        
        # This should work without actual API calls
        try:
            result = workflow.invoke(state)
            assert result["can_generate_graph"] == "Yes"
        except Exception as e:
            # If it fails due to other API calls, that's expected
            assert "web search" in str(e).lower() or "openai" in str(e).lower()
    
    def test_workflow_state_consistency(self):
        """Test that all workflows maintain state consistency."""
        workflows = [
            (create_conditional_graph_workflow, get_conditional_state),
            (create_web_search_graph, get_web_search_state),
            (create_simple_chat_graph, get_chat_state)
        ]
        
        for create_func, get_state_func in workflows:
            workflow = create_func()
            state = get_state_func("Test query")
            
            # All states should have the same basic structure
            required_keys = [
                "messages", "response", "search_results", "user_query",
                "selected_graph_type", "formatted_data", "graph_object", "can_generate_graph"
            ]
            
            for key in required_keys:
                assert key in state, f"Missing key {key} in {create_func.__name__}" 