from openai import OpenAI
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from src.logger import get_logger

logger = get_logger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]
    selected_graph_type: Annotated[str, "The selected graph type"]
    formatted_data: Annotated[str, "The formatted data"]
    graph_object: Annotated[str, "The graph object"]
    can_generate_graph: Annotated[str, "Whether the query can generate a graph (Yes/No)"]


def web_search_node(state: GraphState) -> GraphState:
    """
    Perform web search for the user query using OpenAI's native web search.
    """
    user_query = state["user_query"]
    logger.info(f"Performing web search for: {user_query}")
    
    try:
        # Initialize OpenAI client when needed
        client = OpenAI()
        
        # Use OpenAI's native web search functionality
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=user_query
        )
        
        # Extract the response text
        search_results = response.output_text if hasattr(response, 'output_text') else "No search results found for this query."
        logger.info(f"Raw Search results: {search_results}")
        logger.info(f"OpenAI web search completed successfully")
        
    except Exception as e:
        logger.error(f"OpenAI web search failed: {e}")
        search_results = f"Web search failed: {str(e)}"
    
    return {
        "messages": state["messages"],
        "response": state["response"],
        "search_results": search_results,
        "user_query": user_query,
        "selected_graph_type": state.get("selected_graph_type", ""),
        "formatted_data": state.get("formatted_data", ""),
        "graph_object": state.get("graph_object", None),
        "can_generate_graph": state.get("can_generate_graph", "No")
    } 