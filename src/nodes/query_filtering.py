import json
import os
import yaml
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils import get_llm
from src.logger import get_logger

logger = get_logger(__name__)


def load_classification_prompt():
    """Load the graph classification prompt from file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "graph-classification-instructions.txt")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Classification prompt file not found at {prompt_path}")
        return ""


def get_gpt_mini_config():
    """Load gpt-mini configuration from llm_config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "llm_config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            gpt_mini_config = config.get("gpt-mini", {})
            
            # Ensure model_kwargs exists, even if empty
            if "model_kwargs" not in gpt_mini_config:
                gpt_mini_config["model_kwargs"] = {}
                
            return gpt_mini_config
    except FileNotFoundError:
        logger.error(f"llm_config.yaml not found at {config_path}")
        return {"model_id": "gpt-3.5-turbo", "model_kwargs": {"temperature": 0}}


def query_filtering_node(state):
    """
    Classify whether the user query can generate a graph from web data.
    Returns Yes/No only.
    """
    user_query = state["user_query"]
    
    # Load the classification prompt
    system_prompt = load_classification_prompt()
    
    # Get gpt-mini configuration from config file
    gpt_mini_config = get_gpt_mini_config()
    model_id = gpt_mini_config.get("model_id", "gpt-3.5-turbo")
    model_kwargs = gpt_mini_config.get("model_kwargs", {"temperature": 0})
    
    # Get the LLM with gpt-mini configuration
    llm = get_llm(model_id=model_id, model_kwargs=model_kwargs)
    
    # Create messages for classification
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {user_query}")
    ]
    
    try:
        # Get classification response
        response = llm.invoke(messages)
        response_content = str(response.content).strip()
        
        # Parse the JSON response
        try:
            classification_result = json.loads(response_content)
            can_generate = classification_result.get("can_generate_graph", "No")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response_content}")
            can_generate = "No"
        
        logger.info(f"Query classification: {can_generate}")
        
    except Exception as e:
        logger.error(f"Error in graph classification: {e}")
        can_generate = "No"
    
    # Return state with only the new field added
    return {
        **state,
        "can_generate_graph": can_generate
    }
