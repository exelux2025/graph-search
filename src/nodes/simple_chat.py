import logging
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from src.utils import get_llm

logger = logging.getLogger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]


def chat_node(state: GraphState) -> GraphState:
    """
    Process the conversation and generate a response.
    """
    llm = get_llm()
    messages = state["messages"]
    
    # Invoke the LLM
    response = llm.invoke(messages)
    
    logger.info("Generated simple chat response")
    
    # Update the state
    return {
        "messages": messages + [response],
        "response": str(response.content),
        "search_results": state.get("search_results", ""),
        "user_query": state.get("user_query", "")
    } 