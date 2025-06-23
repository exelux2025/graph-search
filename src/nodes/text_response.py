from src.logger import get_logger

logger = get_logger(__name__)


def text_response_node(state):
    """
    Generate a text response when graphs cannot be generated.
    """
    user_query = state["user_query"]
    
    response_text = f"Graph is not possible for your query: '{user_query}'. Please provide a query that asks for numerical data, comparisons, or statistics that can be visualized in a chart or graph."
    
    logger.info("Generated text response for non-graphable query")
    
    return {
        **state,
        "response": response_text
    } 