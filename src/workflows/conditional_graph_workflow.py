from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage

# Import logger
from src.logger import get_logger

# Import nodes
from src.nodes.web_search import web_search_node, GraphState
from src.nodes.web_search_context import chat_with_search_node
from src.nodes.query_filtering import query_filtering_node
from src.nodes.text_response import text_response_node
from src.nodes.graph_selector import graph_selector_node
from src.nodes.graph_renderer import graph_renderer_node

logger = get_logger(__name__)


def create_conditional_graph_workflow():
    """
    Create a conditional graph workflow that first checks if a query can generate a graph.
    
    Usage:
    - This workflow starts with query filtering to determine if the user query can generate a graph
    - If the query can generate a graph, it proceeds through the full graph generation pipeline
    - If the query cannot generate a graph, it returns a text response asking for a better query
    
    Workflow:
    1. query_filtering -> classifies if query can generate a graph
    2. If "No" -> text_response -> END
    3. If "Yes" -> web_search -> chat_with_search -> graph_selector -> graph_renderer -> END
    """
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("query_filtering", query_filtering_node)
    workflow.add_node("text_response", text_response_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("chat_with_search", chat_with_search_node)
    workflow.add_node("graph_selector", graph_selector_node)
    workflow.add_node("graph_renderer", graph_renderer_node)
    
    # Set the entry point
    workflow.set_entry_point("query_filtering")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "query_filtering",
        lambda state: "text_response" if state["can_generate_graph"] == "No" else "web_search"
    )
    
    # Add edges for graph generation path
    workflow.add_edge("web_search", "chat_with_search")
    workflow.add_edge("chat_with_search", "graph_selector")
    workflow.add_edge("graph_selector", "graph_renderer")
    workflow.add_edge("graph_renderer", END)
    
    # Add edge for text response path
    workflow.add_edge("text_response", END)
    
    # Compile the graph
    return workflow.compile()


def get_initial_state(user_query: str):
    """
    Get the initial state for the conditional graph workflow.
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