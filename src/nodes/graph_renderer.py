import logging
import plotly.graph_objects as go
import plotly.io as pio
from typing import TypedDict, Annotated, List, Any
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]
    selected_graph_type: Annotated[str, "The selected graph type"]
    formatted_data: Annotated[str, "Formatted data for graphing"]
    graph_object: Annotated[Any, "The rendered graph object"]


def parse_data_for_graph(formatted_data: str):
    """
    Simple parser to extract data from formatted string.
    Assumes format: column names on first line, data on subsequent lines.
    """
    lines = formatted_data.strip().split('\n')
    if len(lines) < 2:
        return None, None
    
    # Extract column names (first line)
    columns = [col.strip() for col in lines[0].split(',')]
    
    # Extract data rows
    data_rows = []
    for line in lines[1:]:
        if line.strip():
            row = [val.strip() for val in line.split(',')]
            data_rows.append(row)
    
    return columns, data_rows


def create_graph(graph_type: str, formatted_data: str, user_query: str):
    """
    Create a Plotly graph based on the graph type and data.
    """
    # Set a dark theme for Plotly
    pio.templates.default = "plotly_dark"
    
    columns, data_rows = parse_data_for_graph(formatted_data)
    
    if not columns or not data_rows:
        # Fallback: create a simple text display
        fig = go.Figure()
        fig.add_annotation(
            text="No structured data found for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=user_query, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    try:
        if graph_type == "bar_graph":
            # Extract first two columns for x and y
            x_values = [row[0] for row in data_rows if len(row) > 0]
            y_values = [float(row[1]) if len(row) > 1 and row[1].replace('.', '').replace('-', '').isdigit() else 0 for row in data_rows]
            
            fig = go.Figure(data=[
                go.Bar(x=x_values, y=y_values)
            ])
            fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
        elif graph_type == "pie_chart":
            # Extract first two columns for labels and values
            labels = [row[0] for row in data_rows if len(row) > 0]
            values = [float(row[1]) if len(row) > 1 and row[1].replace('.', '').replace('-', '').isdigit() else 0 for row in data_rows]
            
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values)
            ])
            fig.update_layout(title=user_query, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
        elif graph_type == "line_graph":
            # Extract first two columns for x and y
            x_values = [row[0] for row in data_rows if len(row) > 0]
            y_values = [float(row[1]) if len(row) > 1 and row[1].replace('.', '').replace('-', '').isdigit() else 0 for row in data_rows]
            
            fig = go.Figure(data=[
                go.Scatter(x=x_values, y=y_values, mode='lines+markers')
            ])
            fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
        else:
            # Default to bar graph
            x_values = [row[0] for row in data_rows if len(row) > 0]
            y_values = [float(row[1]) if len(row) > 1 and row[1].replace('.', '').replace('-', '').isdigit() else 0 for row in data_rows]
            
            fig = go.Figure(data=[
                go.Bar(x=x_values, y=y_values)
            ])
            fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating graph: {e}")
        # Fallback: create a simple text display
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating {graph_type}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=user_query, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig


def graph_renderer_node(state: GraphState) -> GraphState:
    """
    Render the graph using Plotly based on the selected graph type and formatted data.
    """
    graph_type = state["selected_graph_type"]
    formatted_data = state["formatted_data"]
    user_query = state["user_query"]
    
    # Create the graph
    graph_object = create_graph(graph_type, formatted_data, user_query)
    
    logger.info(f"Rendered {graph_type} graph")
    
    return {
        "messages": state["messages"],
        "response": state["response"],
        "search_results": state["search_results"],
        "user_query": user_query,
        "selected_graph_type": graph_type,
        "formatted_data": formatted_data,
        "graph_object": graph_object
    } 