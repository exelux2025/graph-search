from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage

# Import logger
from src.logger import get_logger

# Import nodes
from src.nodes.web_search import web_search_node, GraphState
from src.nodes.web_search_context import chat_with_search_node

logger = get_logger(__name__)


def create_web_search_graph():
    """
    Create a graph that performs web search and then generates a response.
    
    Usage:
    - This workflow performs web search for the user query and then processes the results
    - It's useful when you want to get information from the web without generating graphs
    - The workflow is simpler than the conditional graph workflow as it doesn't include graph generation
    
    Workflow:
    1. web_search -> performs web search for the query
    2. chat_with_search -> processes search results and generates a response
    3. END
    """
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("chat_with_search", chat_with_search_node)
    
    # Set the entry point
    workflow.set_entry_point("web_search")
    
    # Add edges
    workflow.add_edge("web_search", "chat_with_search")
    workflow.add_edge("chat_with_search", END)
    
    # Compile the graph
    return workflow.compile()


def get_initial_state(user_query: str):
    """
    Get the initial state for the web search workflow.
    """
    return {
        "messages": [
            SystemMessage(content="You are a helpful assistant that provides information based on web search results.")
        ],
        "response": "",
        "search_results": "",
        "user_query": user_query,
        "selected_graph_type": "",
        "formatted_data": "",
        "graph_object": None,
        "can_generate_graph": ""
    } 