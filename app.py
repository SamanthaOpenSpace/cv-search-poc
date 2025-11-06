import streamlit as st
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
SRC_PATH = APP_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from src.cvsearch.settings import Settings
    from src.cvsearch.api_client import OpenAIClient
    from src.cvsearch.storage import CVDatabase
    from src.cvsearch.search_processor import SearchProcessor
    from src.cvsearch.planner import Planner
    from src.cvsearch.lexicons import (
        load_role_lexicon,
        load_tech_synonyms,
        load_domain_lexicon,
    )
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
def load_stateless_services():
    """
    Initializes all STATELESS services and lexicons *once* and caches them.
    The DB connection and processor will be created on-demand in each page.
    """
    print("--- [Cache Miss] Loading stateless services... ---")

    settings = Settings()

    client = OpenAIClient(settings)
    planner = Planner()

    lexicon_dir = settings.lexicon_dir
    role_lex = load_role_lexicon(lexicon_dir)
    tech_lex = load_tech_synonyms(lexicon_dir)
    domain_lex = load_domain_lexicon(lexicon_dir)

    print("--- Stateless services loaded and cached. ---")

    return {
        "settings": settings,
        "client": client,
        "planner": planner,
        "role_lex": role_lex,
        "tech_lex": tech_lex,
        "domain_lex": domain_lex
    }

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
    col1.metric("Role Lexicons", len(st.session_state.role_lex))
    col2.metric("Tech Lexicons", len(st.session_state.tech_lex))
    col3.metric("Domain Lexicons", len(st.session_state.domain_lex))

except Exception as e:
    st.error(f"Failed to load system status. Error: {e}")

st.info("Go to the **Admin & Ingest** page to load CVs or **Project Search** to find candidates.", icon="ℹ️")