from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage

# Import logger
from src.logger import get_logger

# Import nodes
from src.nodes.simple_chat import chat_node, GraphState

logger = get_logger(__name__)


def create_simple_chat_graph():
    """
    Create a simple chat graph using LangGraph and our LLM utilities.
    
    Usage:
    - This workflow provides a simple chat interface without web search or graph generation
    - It's useful for general conversation and questions that don't require external data
    - The workflow is the simplest of the three, using only the chat node
    
    Workflow:
    1. chat -> processes the conversation and generates a response
    2. END
    """
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add the chat node
    workflow.add_node("chat", chat_node)
    
    # Set the entry point
    workflow.set_entry_point("chat")
    
    # Set the end point
    workflow.add_edge("chat", END)
    
    # Compile the graph
    return workflow.compile()


def get_initial_state(user_query: str):
    """
    Get the initial state for the simple chat workflow.
    """
    return {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            SystemMessage(content=user_query)
        ],
        "response": "",
        "search_results": "",
        "user_query": user_query,
        "selected_graph_type": "",
        "formatted_data": "",
        "graph_object": None,
        "can_generate_graph": ""
    } 