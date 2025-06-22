import logging
from typing import TypedDict, Annotated, List, Any
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]
    selected_graph_type: Annotated[str, "The selected graph type"]
    formatted_data: Annotated[str, "Formatted data for graphing"]
    graph_object: Annotated[Any, "The rendered graph object"]


def format_graph_node(state: GraphState) -> GraphState:
    """
    Format the search results data for graphing.
    """
    search_results = state["search_results"]
    selected_graph_type = state["selected_graph_type"]
    
    # Simple formatting - extract the structured data from search results
    # This assumes the search results contain comma-separated data as specified in the instructions
    
    # For now, just pass through the search results as formatted data
    # In a more complex implementation, this would parse and structure the data
    formatted_data = search_results
    
    logger.info(f"Formatted data for {selected_graph_type} graph")
    
    return {
        "messages": state["messages"],
        "response": state["response"],
        "search_results": search_results,
        "user_query": state["user_query"],
        "selected_graph_type": selected_graph_type,
        "formatted_data": formatted_data,
        "graph_object": state.get("graph_object", None)
    } 