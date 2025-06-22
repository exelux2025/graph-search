import os
import logging
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List

# Import from our modular structure
from src.utils import get_llm
from src.nodes.web_search import web_search_node, GraphState
from src.nodes.web_search_context import chat_with_search_node
from src.nodes.simple_chat import chat_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def create_web_search_graph():
    """
    Create a graph that performs web search and then generates a response.
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


def create_simple_chat_graph():
    """
    Create a simple chat graph using LangGraph and our LLM utilities.
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


def main():
    """
    Main function to demonstrate the LLM utilities.
    """
    logger.info("=== LLM Utilities Demo ===")
    
    # Test basic LLM functionality
    logger.info("\n1. Testing basic LLM functionality:")
    llm = get_llm()
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    response = llm.invoke(messages)
    logger.info(f"Response: {response.content}")
    
    # Test LangGraph integration
    logger.info("\n2. Testing LangGraph integration:")
    graph = create_simple_chat_graph()
    
    # Create initial state
    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me a short joke.")
        ],
        "response": "",
        "search_results": "",
        "user_query": ""
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    logger.info(f"Graph Response: {result['response']}")
    
    # Test web search functionality
    logger.info("\n3. Testing web search functionality:")
    web_search_graph = create_web_search_graph()
    
    # Create initial state for web search
    web_search_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant that provides information based on web search results.")
        ],
        "response": "",
        "search_results": "",
        "user_query": "Tell me the top 10 countries by defense budget in USD"
    }
    
    # Run the web search graph
    web_result = web_search_graph.invoke(web_search_state)
    logger.info(f"Web Search Response: {web_result['response']}")
    
    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
