import logging
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


def chat_with_search_node(state: GraphState) -> GraphState:
    """
    Generate response using LLM with web search results as context.
    """
    llm = get_llm()
    messages = state["messages"]
    search_results = state["search_results"]
    user_query = state["user_query"]
    
    # Create enhanced prompt with search results
    enhanced_prompt = f"""
You are a helpful assistant that provides accurate information based on web search results.

User Query: {user_query}

{search_results}

Based on the web search results above, please provide a comprehensive and accurate answer to the user's query. If the search results contain relevant information, use it to support your response. If the search results are not relevant or insufficient, acknowledge this and provide the best information you can based on your training data.

Please structure your response clearly and cite information from the search results when appropriate.
"""
    
    # Add the enhanced prompt to messages
    enhanced_messages = messages + [HumanMessage(content=enhanced_prompt)]
    
    # Get response from LLM
    response = llm.invoke(enhanced_messages)
    
    logger.info("Generated response with web search context")
    
    return {
        "messages": messages + [response],
        "response": str(response.content),
        "search_results": search_results,
        "user_query": user_query
    } 