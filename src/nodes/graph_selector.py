import logging
import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from src.utils import get_llm

logger = logging.getLogger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]
    selected_graph_type: Annotated[str, "The selected graph type"]
    formatted_data: Annotated[str, "Formatted data for graphing"]
    graph_object: Annotated[str, "The graph object"]


def load_graph_selection_instructions():
    """
    Load graph selection instructions from the prompts file.
    """
    instructions_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "graph-selection-instructions.txt")
    try:
        with open(instructions_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Graph selection instructions file not found at {instructions_path}")
        return "Select the best graph type for this data: {data}"


def graph_selector_node(state: GraphState) -> GraphState:
    """
    Select the best graph type based on the formatted data using OpenAI.
    """
    llm = get_llm()
    formatted_data = state["formatted_data"]
    user_query = state["user_query"]
    
    # Load instructions and replace placeholders
    instructions_template = load_graph_selection_instructions()
    prompt = instructions_template.format(
        data=formatted_data,
        user_query=user_query
    )
    
    # Get graph type selection from LLM
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    # Extract the selected graph type (should be just the graph type name)
    selected_graph_type = str(response.content).strip().lower()
    
    logger.info(f"Selected graph type: {selected_graph_type}")
    
    return {
        "messages": state["messages"],
        "response": state["response"],
        "search_results": state["search_results"],
        "user_query": user_query,
        "selected_graph_type": selected_graph_type,
        "formatted_data": formatted_data,
        "graph_object": state.get("graph_object", None)
    } 