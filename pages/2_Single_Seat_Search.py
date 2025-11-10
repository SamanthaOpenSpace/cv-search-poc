import streamlit as st
import sys
from pathlib import Path
from typing import List # <-- Added List for type hints

APP_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = APP_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from cv_search.clients.openai_client import OpenAIClient
    from cv_search.config.settings import Settings
    from cv_search.db.database import CVDatabase
    from cv_search.search import SearchProcessor
except ImportError as e:
    st.error(f"""
    **Failed to import project modules.**
    
    Ensure `app.py` is running from the project root and the `src` directory is correct.
    **Error:** {e}
    """)
    st.stop()


st.set_page_config(
    page_title="Single Seat Search",
    page_icon="🎯",
    layout="wide"
)
st.title("🎯 Single Seat Search")
st.info("Build a single-seat query using the widgets below. This is ideal for quick tests and debugging.")

if "services_loaded" not in st.session_state:
    st.warning("Services not loaded. Please run the Home page (app.py) first.")
    st.stop()

try:
    settings: Settings = st.session_state.settings
    client: OpenAIClient = st.session_state.client

    # Load the new list-based lexicons
    role_lex_list: List[str] = st.session_state.role_lex
    tech_lex_list: List[str] = st.session_state.tech_lex
    domain_lex_list: List[str] = st.session_state.domain_lex
except KeyError as e:
    st.error(f"Failed to load service or lexicon: {e}. Please return to the Home page and reload.")
    st.stop()


with st.form("seat_query_form"):
    st.subheader("Role Definition")
    col1, col2 = st.columns(2)
    with col1:
        role = st.selectbox(
            "Role",
            # Use the list directly
            options=[""] + sorted(role_lex_list),
            help="The primary role for this seat."
        )
    with col2:
        default_seniority_index = 2
        seniority = st.selectbox(
            "Seniority",
            options=["junior", "middle", "senior", "lead", "manager"],
            index=default_seniority_index
        )

    st.subheader("Technical Skills & Domains")
    must_have = st.multiselect(
        "Must-Have Tech",
        # Use the list directly
        options=sorted(tech_lex_list),
        help="Hard requirements. Candidates will be ranked on matching these."
    )
    nice_to_have = st.multiselect(
        "Nice-to-Have Tech",
        # Use the list directly
        options=sorted(tech_lex_list),
        help="Optional skills that add to the score."
    )
    domains = st.multiselect(
        "Domains",
        # Use the list directly
        options=sorted(domain_lex_list),
        help="Optional domain experience (e.g., 'fintech', 'healthtech')."
    )

    st.subheader("Search Controls")
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        mode = st.radio(
            "Mode",
            options=["hybrid", "lexical", "semantic"],
            index=0,
            horizontal=True
        )
    with control_col2:
        top_k = st.slider("Top-K", min_value=1, max_value=10, value=3)
    with control_col3:
        with_justification = st.toggle("Run LLM Justification", value=True)

    submitted = st.form_submit_button("Search for Seat")


if submitted:
    if not role:
        st.warning("Please select a role.")
        st.stop()

    tech_stack = list(set(must_have + nice_to_have))
    seat_payload = {
        "role": role,
        "seniority": seniority,
        "domains": domains,
        "tech_tags": must_have,
        "nice_to_have": nice_to_have,
        "rationale": "Query built from Single Seat Search UI"
    }
    criteria = {
        "domain": domains,
        "tech_stack": tech_stack,
        "expert_roles": [role],
        "project_type": "greenfield",
        "team_size": {
            "total": 1,
            "members": [seat_payload]
        }
    }

    db = None
    payload = None
    try:
        db = CVDatabase(settings)

        processor = SearchProcessor(db, client, settings)

        with st.spinner("Searching for candidates..."):
            payload = processor.search_for_seat(
                criteria=criteria,
                top_k=top_k,
                run_dir=None,
                mode_override=mode,
                vs_topk_override=None,
                with_justification=with_justification
            )

        if payload:
            st.subheader("Search Results")

            if not payload.get("results"):
                st.warning("No matching candidates found for this query.")

            for result in payload.get("results", []):
                cid = result['candidate_id']
                score = result['score']['value']

                with st.expander(f"**{cid}** (Hybrid Score: {score:.3f})"):

                    justification = result.get('llm_justification')
                    if justification:
                        st.markdown(f"##### ✅ Justification")
                        st.markdown(f"**{justification.get('match_summary')}**")

                        st.markdown("**Strengths:**")
                        for point in justification.get('strength_analysis', []):
                            st.markdown(f"- {point}")

                        st.markdown("**Gaps:**")
                        for point in justification.get('gap_analysis', []):
                            st.markdown(f"- {point}")

                        llm_score = justification.get('overall_match_score', 0.0)
                        st.metric("LLM Match Score", f"{llm_score * 100:.0f}%")

                    else:
                        st.info("Justification was not run for this candidate.")

                    with st.expander("Show CV Context (Evidence)"):
                        context = db.get_full_candidate_context(cid)
                        if context:
                            st.markdown("##### Summary")
                            st.markdown(f"> {context.get('summary_text', 'N/A')}")

                            st.markdown("##### Experience")
                            experience_text = context.get('experience_text', 'N/A')
                            # Split the experience text back into individual job lines
                            # This parsing logic is based on _mk_experience_line in ingestion_pipeline.py
                            experience_lines = experience_text.split(' \n')

                            for line in experience_lines:
                                if not line.strip():
                                    continue

                                parts = line.split(' | ')
                                if parts:
                                    st.markdown(f"**{parts[0]}**") # Title @ Company
                                    with st.container(border=True):
                                        for part in parts[1:]:
                                            if part.startswith("domains: "):
                                                st.markdown(f"**Domains:** `{part[9:]}`")
                                            elif part.startswith("tech: "):
                                                st.markdown(f"**Tech:** `{part[6:]}`")
                                            elif part.startswith("highlights: "):
                                                # Split highlights by " ; "
                                                highlights = part[12:].split(' ; ')
                                                st.markdown("**Highlights:**")
                                                for h in highlights:
                                                    st.markdown(f"- {h}")
                                            else:
                                                st.markdown(part)

                            st.markdown("##### Tags")
                            st.code(context.get('tags_text', 'N/A'), language=None)
                        else:
                            st.error(f"Could not retrieve context for {cid}.")

                    with st.expander("Show Full Result JSON"):
                        st.json(result)

    except Exception as e:
        st.error(f"An error occurred during search: {e}")

    finally:
        if db:
            db.close()