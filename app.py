import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any # <-- Added Dict, Any for type hints

APP_ROOT = Path(__file__).resolve().parent
SRC_PATH = APP_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from cv_search.app.bootstrap import load_stateless_services as bootstrap_stateless_services
    from cv_search.db.database import CVDatabase
    from cv_search.search import SearchProcessor
except ImportError as e:
    st.error(f"""
    **Failed to import project modules.**
    
    Please ensure:
    1. This `app.py` file is in the root of your `cv-search` repository.
    2. The `src` directory exists and contains your project code.
    
    **Error:** {e}
    """)
    st.stop()


@st.cache_resource
def load_stateless_services() -> Dict[str, Any]:
    """
    Initializes all STATELESS services and lexicons *once* and caches them.
    The DB connection and processor will be created on-demand in each page.
    """
    print("--- [Cache Miss] Loading stateless services... ---")

    services: Dict[str, Any] = bootstrap_stateless_services()
    print("--- Stateless services loaded and cached. ---")
    return services

st.set_page_config(
    page_title="CV Search Home",
    page_icon="🏠",
    layout="wide"
)

if "services_loaded" not in st.session_state:
    services = load_stateless_services()
    st.session_state.update(services)
    st.session_state["services_loaded"] = True

st.title("Welcome to CV Search 🏠")
st.markdown("Select a tool from the sidebar to begin.")

st.subheader("System Status")

try:
    col1, col2, col3 = st.columns(3)
    # len() works on both lists and dicts, so this logic is unchanged.
    col1.metric("Role Lexicons", len(st.session_state.role_lex))
    col2.metric("Tech Lexicons", len(st.session_state.tech_lex))
    col3.metric("Domain Lexicons", len(st.session_state.domain_lex))

except Exception as e:
    st.error(f"Failed to load system status. Error: {e}")

st.info("Go to the **Admin & Ingest** page to load CVs or **Project Search** to find candidates.", icon="ℹ️")