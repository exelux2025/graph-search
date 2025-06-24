import logging
import json
import plotly.graph_objects as go
from typing import TypedDict, Annotated, List, Any
from langchain_core.messages import BaseMessage
from src.logger import get_logger
import traceback

logger = get_logger(__name__)

# Define the state for our graph
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    response: Annotated[str, "The response from the LLM"]
    search_results: Annotated[str, "Results from web search"]
    user_query: Annotated[str, "The original user query"]
    selected_graph_type: Annotated[str, "The selected graph type"]
    formatted_data: Annotated[str, "Formatted data for graphing"]
    graph_object: Annotated[Any, "The rendered graph object"]
    can_generate_graph: Annotated[str, "Whether the query can generate a graph (Yes/No)"]
    selected_columns: Annotated[List[str], "The selected columns for graphing"]

def to_float(value: Any) -> float:
    """Safely converts a value to a float, handling strings with commas."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def parse_data_for_graph(formatted_data: str, selected_columns=None):
    """
    Parses a JSON string to extract data for graphing, using only selected columns if provided.
    """
    try:
        formatted_data = formatted_data.strip()
        if formatted_data.startswith("```json"):
            formatted_data = formatted_data[7:-3].strip()
        data = json.loads(formatted_data)
        # Defensive: strip whitespace from all keys
        if any(k.strip() != k for k in data.keys()):
            data = {k.strip(): v for k, v in data.items()}
        # Use only selected columns if provided
        if selected_columns and isinstance(selected_columns, list) and len(selected_columns) >= 2:
            columns = selected_columns
            logger.info(f"Using selected columns for graph: {columns}")
        elif "col_names" in data:
            columns = data.get("col_names", [])
            logger.info(f"Using all columns from col_names: {columns}")
        else:
            columns = list(data.keys())
            logger.info(f"Using all columns from keys: {columns}")
        if not columns or len(columns) < 2:
            return None, None
        # Reconstruct data rows from the column-oriented JSON
        num_rows = len(data.get(columns[0], {}).get("values", []))
        data_rows = []
        for i in range(num_rows):
            row = [data.get(col, {}).get("values", [])[i] for col in columns]
            data_rows.append(row)
        logger.info(f"Parsed data: {len(columns)} columns, {len(data_rows)} rows")
        return columns, data_rows
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.error(f"Error parsing JSON data for graph: {formatted_data!r} | Error: {e}")
        return None, None


def create_graph(graph_type: str, formatted_data: str, user_query: str, selected_columns=None):
    logger.info(f"create_graph called with graph_type={graph_type}, selected_columns={selected_columns}")
    try:
        logger.info("About to parse data for graph")
        columns, data_rows = parse_data_for_graph(formatted_data, selected_columns)
        logger.info(f"parse_data_for_graph returned columns={columns}, data_rows length={len(data_rows) if data_rows else 0}")
        if not columns or not data_rows:
            logger.warning("No structured data found for visualization")
            fig = go.Figure()
            fig.add_annotation(
                text="No structured data found for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=user_query)
            logger.info("Returning from create_graph (no data)")
            return fig
        logger.info(f"Successfully parsed data: {len(columns)} columns, {len(data_rows)} rows")
        logger.info(f"Rendering {graph_type} graph")
        
        try:
            if graph_type == "bar_graph":
                x_values = [str(row[0]) if len(row) > 0 else "" for row in data_rows]
                y_values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
                fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1])
            elif graph_type == "stacked_bar_chart":
                if len(columns) >= 3:
                    categories = [str(row[0]) for row in data_rows]
                    traces = []
                    for col_idx in range(1, len(columns)):
                        series_name = columns[col_idx]
                        y_values = [to_float(row[col_idx]) if len(row) > col_idx else 0.0 for row in data_rows]
                        traces.append(go.Bar(name=series_name, x=categories, y=y_values))
                    fig = go.Figure(data=traces)
                    fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title="Value", barmode='stack')
                else:
                    x_values = [str(row[0]) if len(row) > 0 else "" for row in data_rows]
                    y_values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
                    fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1])
            elif graph_type == "pie_chart":
                labels = [str(row[0]) if len(row) > 0 else "" for row in data_rows]
                values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig.update_layout(title=user_query)
            elif graph_type == "line_graph":
                x_values = [str(row[0]) if len(row) > 0 else "" for row in data_rows]
                y_values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                fig = go.Figure(data=[go.Scatter(x=x_values, y=y_values, mode='lines+markers')])
                fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1])
            elif graph_type == "scatterplot":
                x_values = [to_float(row[0]) if len(row) > 0 else 0.0 for row in data_rows]
                y_values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                fig = go.Figure(data=[go.Scatter(x=x_values, y=y_values, mode='markers')])
                fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1])
            elif graph_type == "multi_bar_graph":
                if len(columns) >= 3:
                    categories = [str(row[0]) for row in data_rows]
                    traces = []
                    for col_idx in range(1, len(columns)):
                        series_name = columns[col_idx]
                        y_values = [to_float(row[col_idx]) if len(row) > col_idx else 0.0 for row in data_rows]
                        traces.append(go.Bar(name=series_name, x=categories, y=y_values))
                    fig = go.Figure(data=traces)
                    fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title="Value", barmode='group')
                else:
                    x_values = [str(row[0]) if len(row) > 0 else "" for row in data_rows]
                    y_values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                    fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
                    fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1])
            else:
                x_values = [str(row[0]) if len(row) > 0 else "" for row in data_rows]
                y_values = [to_float(row[1]) if len(row) > 1 else 0.0 for row in data_rows]
                fig = go.Figure(data=[go.Bar(x=x_values, y=y_values)])
                fig.update_layout(title=user_query, xaxis_title=columns[0], yaxis_title=columns[1])
            logger.info("Returning from create_graph (success)")
            return fig
        except Exception as e:
            logger.error(f"Exception in create_graph inner graphing: {e}\n{traceback.format_exc()}")
            raise
    except Exception as e:
        logger.error(f"Exception in create_graph: {e}\n{traceback.format_exc()}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating {graph_type}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=user_query)
        logger.info("Returning from create_graph (exception fallback)")
        return fig


def graph_renderer_node(state: GraphState) -> GraphState:
    graph_type = state["selected_graph_type"]
    formatted_data = state["formatted_data"]
    user_query = state["user_query"]
    selected_columns = state.get("selected_columns", None)
    logger.info(f"graph_renderer_node called with graph_type={graph_type}, selected_columns={selected_columns}")
    try:
        graph_object = create_graph(graph_type, formatted_data, user_query, selected_columns)
        logger.info(f"Rendered {graph_type} graph")
        logger.info("Returning from graph_renderer_node")
        return {
            "messages": state["messages"],
            "response": state["response"],
            "search_results": state["search_results"],
            "user_query": user_query,
            "selected_graph_type": graph_type,
            "selected_columns": selected_columns,
            "formatted_data": formatted_data,
            "graph_object": graph_object,
            "can_generate_graph": state["can_generate_graph"]
        }
    except Exception as e:
        logger.error(f"Exception in graph_renderer_node: {e}\n{traceback.format_exc()}")
        raise 