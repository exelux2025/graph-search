import os
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
    formatted_data: Annotated[str, "The formatted data"]
    graph_object: Annotated[str, "The graph object"]
    can_generate_graph: Annotated[str, "Whether the query can generate a graph (Yes/No)"]


def load_instructions():
    """
    Load instructions from the web-search-instructions.txt file in src/prompts/.
    """
    instructions_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "web-search-instructions.txt")
    try:
        with open(instructions_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Instructions file not found at {instructions_path}")
        return "You are a helpful assistant. Please provide a comprehensive answer to the user's query."


def chat_with_search_node(state: GraphState) -> GraphState:
    """
    Cleans and formats web search results into structured data using an LLM.
    """
    llm = get_llm()
    search_results = state["search_results"]
    user_query = state["user_query"]
    
    # Load instructions from file and replace placeholders
    instructions_template = load_instructions()
    enhanced_prompt = instructions_template.format(
        user_query=user_query,
        search_results=search_results
    )
    
    # Use a fresh message list for the LLM call to get only the structured data
    llm_messages = [HumanMessage(content=enhanced_prompt)]
    
    # Get response from LLM - this response should be the cleaned data
    response = llm.invoke(llm_messages)
    
    cleaned_data = str(response.content)
    logger.info("Cleaned search results into structured data.")

    logger.info(f"Cleaned data: {cleaned_data}")    
    
    # The cleaned data is the main "response" to the user, and also the formatted data for the graph
    return {
        "messages": state["messages"], # Pass original messages through
        "response": cleaned_data,
        "search_results": search_results,
        "user_query": user_query,
        "selected_graph_type": state.get("selected_graph_type", ""),
        "formatted_data": cleaned_data,
        "graph_object": state.get("graph_object", None),
        "can_generate_graph": state.get("can_generate_graph", "No")
    } 