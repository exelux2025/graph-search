import streamlit as st
import logging
import sys
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
import json

# Import from our modular structure
from src.utils import get_llm
from src.nodes.web_search import web_search_node, GraphState
from src.nodes.web_search_context import chat_with_search_node
from src.nodes.graph_selector import graph_selector_node
from src.nodes.graph_renderer import graph_renderer_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Graph Search & Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Graph Search & Visualization")
st.markdown("Search for data and automatically generate visualizations!")

def create_full_workflow():
    """
    Create the complete workflow with web search, data cleaning, graph selection, and rendering.
    """
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("data_formatter", chat_with_search_node)
    workflow.add_node("graph_selector", graph_selector_node)
    workflow.add_node("graph_renderer", graph_renderer_node)
    
    # Set entry point
    workflow.set_entry_point("web_search")
    
    # Add edges
    workflow.add_edge("web_search", "data_formatter")
    workflow.add_edge("data_formatter", "graph_selector")
    workflow.add_edge("graph_selector", "graph_renderer")
    workflow.add_edge("graph_renderer", END)
    
    return workflow.compile()

def main():
    # Create the workflow
    graph = create_full_workflow()
    
    # User input
    user_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., top 10 countries by defense budget in USD"
    )
    
    if st.button("🔍 Search & Visualize"):
        if user_query:
            with st.spinner("Processing your request..."):
                try:
                    # Create initial state
                    initial_state = {
                        "messages": [
                            SystemMessage(content="You are a helpful assistant that provides information based on web search results.")
                        ],
                        "response": "",
                        "search_results": "",
                        "user_query": user_query,
                        "selected_graph_type": "",
                        "formatted_data": "",
                        "graph_object": None
                    }
                    
                    # Run the workflow
                    result = graph.invoke(initial_state)
                    
                    # Display results
                    st.success("✅ Processing complete!")
                    
                    # Show the text response
                    st.subheader("📝 Text Response")
                    st.write(result["response"])
                    
                    # Show the selected graph type
                    st.subheader("📈 Selected Visualization")
                    st.info(f"Graph Type: {result['selected_graph_type']}")
                    
                    # Show the graph
                    st.subheader("📊 Generated Graph")
                    if result["graph_object"]:
                        st.plotly_chart(result["graph_object"], use_container_width=True)
                    else:
                        st.error("No graph was generated.")
                    
                    # Show raw data for debugging
                    with st.expander("🔍 Raw Data"):
                        st.text("Raw Web Search Results:")
                        st.text(result["search_results"])
                        st.subheader("Formatted JSON Data")
                        try:
                            # Sanitize and load the JSON string for display
                            json_string = result["formatted_data"]
                            if json_string.strip().startswith("```json"):
                                json_string = json_string.strip()[7:-3].strip()
                            st.json(json.loads(json_string))
                        except Exception:
                            st.text("Could not parse formatted data as JSON. Displaying raw string:")
                            st.text(result["formatted_data"])
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Streamlit app error: {e}")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main() 