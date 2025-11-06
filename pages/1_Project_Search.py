import streamlit as st
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = APP_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from src.cvsearch.storage import CVDatabase
    from src.cvsearch.search_processor import SearchProcessor
    from src.cvsearch.planner import Planner
    from src.cvsearch.api_client import OpenAIClient
    from src.cvsearch.settings import Settings
    from src.cvsearch.parser import parse_request, Criteria, TeamSize, TeamMember
except ImportError as e:
    st.error(f"""
    **Failed to import project modules.**
    
    Ensure `app.py` is running from the project root and the `src` directory is correct.
    **Error:** {e}
    """)
    st.stop()


st.set_page_config(
    page_title="Project Search",
    page_icon="🚀",
    layout="wide"
)
st.title("🚀 Project Search & Planning")

if "services_loaded" not in st.session_state:
    st.warning("Services not loaded. Please run the Home page (app.py) first.")
    st.stop()

try:
    planner: Planner = st.session_state.planner
    settings: Settings = st.session_state.settings
    client: OpenAIClient = st.session_state.client
except KeyError as e:
    st.error(f"Failed to load service: {e}. Please return to the Home page and reload.")
    st.stop()


search_tab, plan_tab = st.tabs(["Search for Project", "Plan Presale Team"])

with search_tab:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option 1: Search by Text Brief")
        text_brief = st.text_area(
            "Enter a free-text project brief:",
            height=200,
            placeholder="e.g., 'We need two senior .NET developers and one React dev for a fintech project...'"
        )

    with col2:
        st.subheader("Option 2: Search by Criteria File")
        criteria_file = st.file_uploader(
            "Upload a criteria.json file",
            type=['json']
        )

    st.divider()

    search_cols = st.columns(2)
    with search_cols[0]:
        top_k = st.slider(
            "Top-K candidates per seat:",
            min_value=1, max_value=10, value=3
        )
    with search_cols[1]:
        with_justification = st.toggle(
            "Run LLM Justification (Slower)",
            value=True
        )

    if st.button("Run Project Search", type="primary"):
        payload = None
        raw_text = None
        crit_obj = None

        if criteria_file:
            try:
                crit_dict = json.load(criteria_file)
                ts_dict = crit_dict.get("team_size", {})
                members = [
                    TeamMember(
                        role=m["role"], seniority=m.get("seniority"), domains=m.get("domains", []),
                        tech_tags=m.get("tech_tags", []), nice_to_have=m.get("nice_to_have", []),
                        rationale=m.get("rationale")
                    ) for m in ts_dict.get("members", [])
                ]
                team_size_obj = TeamSize(total=ts_dict.get("total"), members=members)
                crit_obj = Criteria(
                    domain=crit_dict.get("domain", []), tech_stack=crit_dict.get("tech_stack", []),
                    expert_roles=crit_dict.get("expert_roles", []),
                    project_type=crit_dict.get("project_type"), team_size=team_size_obj
                )
                st.success("Searching based on uploaded JSON criteria...")

            except Exception as e:
                st.error(f"Error parsing criteria.json: {e}")

        elif text_brief:
            st.success("Parsing text brief and searching...")
            raw_text = text_brief
            crit_obj = parse_request(
                raw_text, settings.openai_model, settings, client
            )

        else:
            st.warning("Please provide a text brief or upload a criteria file.")
            st.stop()

        if crit_obj:

            db = None
            try:
                # Open the database connection. It will be used for both
                # the search and fetching context in the results loop.
                db = CVDatabase(settings)

                processor = SearchProcessor(db, client, settings)

                with st.spinner("Searching... (Justification may take a moment)"):
                    payload = processor.search_for_project(
                        criteria=crit_obj,
                        top_k=top_k,
                        run_dir=None,
                        raw_text=raw_text,
                        with_justification=with_justification
                    )

                if not payload:
                    st.error("Search failed to return a payload.")
                    st.stop()

                st.subheader("Search Results")

                if raw_text:
                    with st.expander("Show Derived Criteria"):
                        st.json(payload.get("project_criteria"))

                if payload.get("gaps"):
                    st.warning(f"Could not find candidates for seat(s): {payload['gaps']}")

                seat_tabs = st.tabs([
                    f"Seat {s['index']}: {s['role']}" for s in payload['seats']
                ])

                for i, seat_data in enumerate(payload['seats']):
                    with seat_tabs[i]:
                        st.write(f"**Role:** {seat_data['role']}")
                        seat_rationale = "No rationale provided."
                        try:
                            seat_rationale = seat_data['criteria']['team_size']['members'][0]['rationale']
                        except (KeyError, IndexError):
                            pass
                        st.write(f"**Rationale:** *{seat_rationale}*")

                        if not seat_data['results']:
                            st.write("No matching candidates found.")
                            continue

                        for result in seat_data['results']:
                            cid = result['candidate_id']
                            score = result['score']['value']

                            with st.expander(f"**{cid}** (Score: {score:.3f})"):
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
                                    score = justification.get('overall_match_score', 0.0)
                                    st.metric("LLM Match Score", f"{score * 100:.0f}%")
                                else:
                                    st.info("Justification was not run for this candidate.")

                                # --- UPDATED CODE BLOCK ---
                                with st.expander("Show CV Context (Evidence)"):
                                    # Use the existing 'db' connection
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
                                # --- END UPDATED CODE BLOCK ---

                                with st.expander("Show Full Result JSON"):
                                    st.json(result)

            except Exception as e:
                st.error(f"An error occurred during search: {e}")

            finally:
                # The 'finally' block now correctly closes the connection
                # after the search AND the results rendering are complete.
                if db:
                    db.close()

with plan_tab:
    st.subheader("Generate Presale Team")
    st.info("This tool generates a budget-agnostic presale team "
            "based *only* on the tech and features in the brief. (This feature is currently disabled)")

    presale_text = st.text_area(
        "Enter a project brief to plan for:",
        height=200,
        placeholder="e.g., 'Mobile + web app with Flutter/React; AI chatbot for goal setting...'",
        disabled=True
    )

    if st.button("Generate Presale Plan", disabled=True):
        pass
        # if not presale_text:
        #     st.warning("Please enter a brief.")
        # else:
        #     with st.spinner("Analyzing brief and planning team..."):
        #         try:
        #             crit = parse_request(
        #                 presale_text,
        #                 model=settings.openai_model,
        #                 settings=settings,
        #                 client=client
        #             )
        #
        #             plan = planner.derive_presale_team(
        #                 crit, raw_text=presale_text
        #             )
        #
        #             st.subheader("Recommended Presale Team")
        #             st.json(plan)
        #
        #         except Exception as e:
        #             st.error(f"Error during planning: {e}")