import os
import yaml
import logging
import sys
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Default LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

LLM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_config.yaml")


class LLMProvisioner:
    """
    Factory/provider for singleton, LangGraph-compatible LLM objects.
    Supports OpenAI with configurable models and parameters.
    Loads config from YAML and caches the LLM instance.
    """

    _llm_instance = None
    _llm_config = None

    @classmethod
    def get_llm(cls, model_id=None, model_kwargs=None):
        """
        Return a singleton LLM instance compatible with LangGraph.
        Loads config from YAML if not provided.
        """
        if cls._llm_instance is not None:
            return cls._llm_instance

        provider = LLM_PROVIDER

        # Load the LLM configuration from YAML
        config = cls._load_llm_config()

        # Allow override from arguments
        model_id = model_id or config.get("model_id")
        model_kwargs = model_kwargs or config.get("model_kwargs", {})

        logger.info(
            f"Using provider: {provider}, model_id: {model_id}, model_kwargs: {model_kwargs}"
        )
        
        if provider == "openai":
            cls._llm_instance = cls._create_openai_llm(
                model_id=model_id, model_kwargs=model_kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        return cls._llm_instance

    @classmethod
    def _create_openai_llm(cls, model_id, model_kwargs):
        """
        Create a LangChain ChatOpenAI instance.
        """
        temperature = (model_kwargs or {}).get("temperature", 0)
        return ChatOpenAI(model=model_id or "gpt-4o", temperature=temperature)

    @classmethod
    def _load_llm_config(cls):
        """
        Load LLM config from llm_config.yaml based on LLM_PROVIDER env var.
        Caches the config for efficiency.
        """
        if cls._llm_config is not None:
            return cls._llm_config
        with open(LLM_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError(
                f"llm_config.yaml is empty or invalid at {LLM_CONFIG_PATH}"
            )
        provider = LLM_PROVIDER
        if provider not in config:
            raise ValueError(f"LLM provider '{provider}' not found in llm_config.yaml")
        cls._llm_config = config[provider]
        return cls._llm_config


# Convenience function for legacy code
def get_llm(model_id=None, model_kwargs=None):
    """
    Return a singleton, LangGraph-compatible LLM instance (OpenAI).
    """
    return LLMProvisioner.get_llm(model_id=model_id, model_kwargs=model_kwargs)
