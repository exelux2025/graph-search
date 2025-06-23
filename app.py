import streamlit as st
from dotenv import load_dotenv
import json

# Import logger
from src.logger import get_logger

# Import workflows
from src.workflows.conditional_graph_workflow import create_conditional_graph_workflow, get_initial_state as get_conditional_state
from src.workflows.web_search_workflow import create_web_search_graph, get_initial_state as get_web_search_state
from src.workflows.simple_chat_workflow import create_simple_chat_graph, get_initial_state as get_chat_state

# Get logger
logger = get_logger(__name__)

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

def main():
    # Workflow selection
    st.sidebar.header("🔧 Workflow Selection")
    workflow_type = st.sidebar.selectbox(
        "Choose a workflow:",
        [
            "Conditional Graph Workflow",
            "Web Search Only",
            "Simple Chat"
        ],
        help="Select the type of processing you want to perform"
    )
    
    # Create the appropriate workflow
    if workflow_type == "Conditional Graph Workflow":
        graph = create_conditional_graph_workflow()
        get_state_func = get_conditional_state
        workflow_description = """
        **Conditional Graph Workflow**: 
        - First checks if your query can generate a graph
        - If yes: performs web search, formats data, selects graph type, and renders visualization
        - If no: provides a helpful text response asking for a better query
        - Best for queries that might or might not be suitable for visualization
        """
    elif workflow_type == "Web Search Only":
        graph = create_web_search_graph()
        get_state_func = get_web_search_state
        workflow_description = """
        **Web Search Only**: 
        - Performs web search for your query
        - Processes and formats the search results
        - Provides a text response without graph generation
        - Best for informational queries that don't need visualization
        """
    else:  # Simple Chat
        graph = create_simple_chat_graph()
        get_state_func = get_chat_state
        workflow_description = """
        **Simple Chat**: 
        - Provides a basic chat interface
        - No web search or graph generation
        - Best for general conversation and questions
        """
    
    # Display workflow description
    st.sidebar.markdown(workflow_description)
    
    # User input
    user_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., top 10 countries by defense budget in USD"
    )
    
    if st.button("🔍 Process Query"):
        if user_query:
            with st.spinner("Processing your request..."):
                try:
                    # Create initial state using the appropriate function
                    initial_state = get_state_func(user_query)
                    
                    # Run the workflow
                    result = graph.invoke(initial_state)
                    
                    # Display results
                    st.success("✅ Processing complete!")
                    
                    # Show the text response
                    st.subheader("📝 Text Response")
                    st.write("Retrieved data ✅")
                    
                    # Show additional information based on workflow type
                    if workflow_type == "Conditional Graph Workflow":
                        # Show classification result
                        st.subheader("🔍 Query Classification")
                        classification_status = "✅ Can generate graph" if result["can_generate_graph"] == "Yes" else "❌ Cannot generate graph"
                        st.info(classification_status)
                        
                        # Show graph if it was generated
                        if result.get("graph_object") and result["can_generate_graph"] == "Yes":
                            st.subheader("📈 Selected Visualization")
                            st.info(f"Graph Type: {result['selected_graph_type']}")
                            
                            st.subheader("📊 Generated Graph")
                            st.plotly_chart(result["graph_object"], use_container_width=True)
                    
                    elif workflow_type == "Web Search Only":
                        # Show search results
                        st.subheader("🔍 Web Search Results")
                        st.text(result["search_results"])
                    
                    # Show raw data for debugging (only for workflows that have it)
                    if result.get("formatted_data") and workflow_type != "Simple Chat":
                        with st.expander("🔍 Raw Data"):
                            if result.get("search_results"):
                                st.text("Raw Web Search Results:")
                                st.text(result["search_results"])
                            
                            if result.get("formatted_data"):
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