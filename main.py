"""Streamlit app for annotating LangSmith datasets."""
import logging
import datetime

import pandas as pd
import streamlit as st
from langsmith import Client

from app.core.config import config as c
from app.core.exceptions import ReadLangsmithDatasetException
from app.core.loggers import setup_logging

setup_logging(c.LOG_LEVEL, c.LOG_USE_BASIC_FORMAT)
_LOGGER = logging.getLogger(__name__)

# Initialize LangSmith client
client = Client()

st.set_page_config(layout="wide")


@st.cache_data
def fetch_dataset(dataset_name: str) -> list[dict]:
    """Fetch dataset from LangSmith.

    Args:
        dataset_name: Name of the dataset to fetch.

    Returns:
        List of dataset records with input, output, and metadata.

    Raises:
        ValueError: If dataset cannot be fetched.
    """
    try:
        dataset = client.list_examples(dataset_name=dataset_name)
        return [
            {
                "id": str(record.id),
                "input": str(record.inputs),  # Convert to string for display
                "output": str(record.outputs),  # Convert to string for display
                "metadata": str(record.metadata),  # Convert to string for display
                "conversation": str(record.inputs.get("input_conversation")),
                "query": str(record.inputs.get("input_query")),
                "intent": str(record.inputs.get("input_intent")),
                "language": str(record.inputs.get("input_language")),
                "persona": str(record.inputs.get("input_persona")),
                "response": str(record.outputs.get("output_response")),
                "extraction": str(record.outputs.get("output_extraction")),
                "task": (
                    "chat" if record.outputs.get("output_response")
                    else "extraction"
                ),
                "human_annotation": "",  # Initialize empty for human annotation
                "remarks": "",
            }
            for record in dataset
        ]
    except Exception as exc:
        raise ReadLangsmithDatasetException(
            f"Error fetching dataset: {str(exc)}"
        ) from exc

def _on_task_change():
    _LOGGER.info("Showing task %s", st.session_state.task_type)

def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("LangSmith Dataset Annotation Tool")

    # Dataset selection
    dataset_name = st.text_input("Enter Dataset Name")

    if dataset_name:
        try:
            data = fetch_dataset(dataset_name)
            df = {
                "extraction": pd.DataFrame(
                    [d for d in data if d["task"] == "extraction"]
                ),
                "chat": pd.DataFrame(
                    [d for d in data if d["task"] == "chat"]
                )
            }
            task_type = st.selectbox(
                "Task type", options=("extraction", "chat"), key="task_type",
                on_change=_on_task_change,
            )

            # Configure columns for data editor
            column_config = {
                "input": st.column_config.JsonColumn(label="inputs"),
                "output": st.column_config.JsonColumn(label="outputs"),
                "metadata": {"disabled": True},
                "human_annotation": {
                    "label": "Your Annotation",
                    "help": "Enter your human annotation here",
                    "required": True
                },
                "remarks": {
                    "label": "Remarks",
                    "help": "Enter your remarks here",
                    "required": False
                }
            }

            st.subheader("Annotation Interface")
            if task_type:
                edited_df = st.data_editor(
                    df[task_type],
                    column_config=column_config,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="data_editor",
                    column_order=(
                        (
                            "id", "persona", "language", "intent",
                            "conversation", "query", "response", "extraction",
                            "human_annotation", "remarks",
                        ) if task_type == "extraction" else (
                            "id", "language", "intent",
                            "conversation", "query", "response",
                            "human_annotation", "remarks",
                        )
                    ),
                    disabled=(
                        c for c in df[task_type].columns
                        if c not in ["human_annotation", "remarks"]
                    ),
                )

            # Save and export buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Annotations"):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{dataset_name}_annotations_{timestamp}.json"
                    edited_df.to_json(filename, orient="records", indent=2)
                    st.success(f"Annotations saved to {filename}!")

            with col2:
                if st.button("Export Annotations"):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{dataset_name}_annotations_{timestamp}.csv"
                    edited_df.to_csv(filename, index=False)
                    st.success(f"Annotations exported to {filename}!")
                    st.download_button(
                        label="Download CSV",
                        data=edited_df.to_csv(index=False).encode('utf-8'),
                        file_name=filename,
                        mime='text/csv',
                    )

        except ValueError as exc:
            st.error(str(exc))
        except ReadLangsmithDatasetException as exc:
            st.error(exc)

if __name__ == "__main__":
    main()
