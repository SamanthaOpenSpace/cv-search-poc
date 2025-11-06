import streamlit as st
import sys
import json
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = APP_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from src.cvsearch.storage import CVDatabase
    from src.cvsearch.api_client import OpenAIClient
    from src.cvsearch.settings import Settings
    from src.cvsearch.ingestion_pipeline import CVIngestionPipeline
    from src.cvsearch.search_processor import SearchProcessor
except ImportError as e:
    st.error(f"""
    **Failed to import project modules.**
    
    Ensure `app.py` is running from the project root and the `src` directory is correct.
    **Error:** {e}
    """)
    st.stop()


st.set_page_config(
    page_title="Admin & Ingest",
    page_icon="📥",
    layout="wide"
)
st.title("📥 Ingestion & System Administration")

if "services_loaded" not in st.session_state:
    st.warning("Services not loaded. Please run the Home page (app.py) first.")
    st.stop()

try:
    settings: Settings = st.session_state.settings
    client: OpenAIClient = st.session_state.client
except KeyError as e:
    st.error(f"Failed to load service: {e}. Please return to the Home page and reload.")
    st.stop()


def get_db_pipeline():
    """Creates a new DB and Pipeline instance for thread safety."""
    db = CVDatabase(settings)
    pipeline = CVIngestionPipeline(db, settings)
    return db, pipeline

st.subheader("Upload New CVs")
st.info("Upload one or more JSON files containing CV data. "
        "Files can contain a single CV object or a list of CV objects "
        "(like mock_cvs.json).")

uploaded_files = st.file_uploader(
    "Upload CV JSON files",
    type=['json'],
    accept_multiple_files=True
)

if st.button("Ingest New CVs"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        db, pipeline = get_db_pipeline()
        cvs_to_ingest = []

        try:
            with st.spinner(f"Reading {len(uploaded_files)} file(s)..."):
                for file_obj in uploaded_files:
                    try:
                        data = json.load(file_obj)

                        if isinstance(data, list):
                            cvs_to_ingest.extend(data)
                        elif isinstance(data, dict):
                            cvs_to_ingest.append(data)
                        else:
                            st.error(f"File '{file_obj.name}' has invalid JSON format.")

                    except json.JSONDecodeError:
                        st.error(f"Error decoding JSON from file: {file_obj.name}")

            if cvs_to_ingest:
                with st.spinner(f"Ingesting {len(cvs_to_ingest)} CV(s). "
                                "This will re-build the FAISS index..."):

                    count = pipeline.run_ingestion_from_list(cvs_to_ingest)

                    # Clear cached services so the next load picks up the new FAISS index.
                    st.cache_resource.clear()
                    st.session_state.clear()

                    st.success(f"Successfully ingested {count} CVs. "
                               "All caches cleared.")
                    st.info("Please refresh the page or navigate to Home.")

        except Exception as e:
            st.error(f"An error occurred during ingestion: {e}")

        finally:
            if db:
                db.close()

st.divider()
st.subheader("Mock Data")

if st.button("Re-ingest All Mock CVs"):
    db, pipeline = get_db_pipeline()
    try:
        with st.spinner("Ingesting mock CVs and rebuilding index..."):
            count = pipeline.run_mock_ingestion()

            st.cache_resource.clear()
            st.session_state.clear()

            st.success(f"Successfully ingested {count} mock CVs. "
                       "All caches cleared.")
            st.info("Please refresh the page or navigate to Home.")

    except Exception as e:
        st.error(f"An error occurred during mock ingestion: {e}")

    finally:
        if db:
            db.close()

st.divider()
st.subheader("System Status")

if st.button("Refresh System Status"):
    db = None
    try:
        db = CVDatabase(settings)
        processor = SearchProcessor(db, client, settings)

        faiss_count = 0
        if processor.semantic_retriever and processor.semantic_retriever.vector_db:
            faiss_count = processor.semantic_retriever.vector_db.ntotal

        tables = db.check_tables()

        col1, col2 = st.columns(2)
        col1.metric("Candidates in Index (FAISS)", faiss_count)
        col2.metric("Database Tables", len(tables))
        st.info(f"Tables found: {', '.join(tables)}")

    except Exception as e:
        st.error(f"Could not read system status. Is the DB initialized? Error: {e}")

    finally:
        if db:
            db.close()