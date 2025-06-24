import os
import json
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from src.utils import get_llm
from src.logger import get_logger

logger = get_logger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]
    selected_graph_type: Annotated[str, "The selected graph type"]
    selected_columns: Annotated[List[str], "The selected columns"]
    formatted_data: Annotated[str, "Formatted data for graphing"]
    graph_object: Annotated[str, "The graph object"]
    can_generate_graph: Annotated[str, "Whether the query can generate a graph (Yes/No)"]


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
    logger.info(f"Graph selector node called with state: {state}")
    formatted_data = state["formatted_data"]
    user_query = state["user_query"]

    # Ensure formatted_data is a minified JSON string
    if isinstance(formatted_data, dict):
        formatted_data_str = json.dumps(formatted_data, separators=(',', ':'))
    else:
        formatted_data_str = str(formatted_data).strip()

    # Load instructions and replace placeholders
    instructions_template = load_graph_selection_instructions()
    prompt = instructions_template.format(
        data=formatted_data_str,
        user_query=user_query
    )
    logger.info(f"Graph selector prompt: {prompt[:500]}...")

    messages = [HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        logger.info(f"LLM response: {response.content!r}")
        # Clean up LLM response
        response_str = str(response.content).strip()
        if response_str.startswith("```json"):
            response_str = response_str[7:-3].strip()
        result_json = json.loads(response_str)
        selected_graph_type = result_json.get("selected_graph_type", "").lower()
        selected_columns = result_json.get("selected_columns", [])
        logger.info(f"Selected graph type: {selected_graph_type}, columns: {selected_columns}")
    except Exception as e:
        logger.error(f"Error during LLM graph selection: {e}\nPrompt sent: {prompt!r}")
        raise

    return {
        "messages": state["messages"],
        "response": state["response"],
        "search_results": state["search_results"],
        "user_query": user_query,
        "selected_graph_type": selected_graph_type,
        "selected_columns": selected_columns,
        "formatted_data": formatted_data,
        "graph_object": state.get("graph_object", None),
        "can_generate_graph": state.get("can_generate_graph", "No")
    } 