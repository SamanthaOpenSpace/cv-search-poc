from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
import re # <-- NEW IMPORT
from collections import defaultdict # <-- NEW IMPORT

# --- NEW IMPORTS for Component 3 ---
import hashlib
import shutil
from datetime import datetime
from src.cvsearch.cv_parser import CVParser
# --- END NEW IMPORTS ---

# --- NEW IMPORT for Parallelism ---
from concurrent.futures import ThreadPoolExecutor, as_completed
# --- END NEW IMPORT ---

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from src.cvsearch.search_processor import SearchProcessor, default_run_dir
from src.cvsearch.planner import Planner
import subprocess  # Import subprocess for error handling
from src.cvsearch.gdrive_sync import GDriveSyncer

import click
from src.cvsearch.storage import CVDatabase
from src.cvsearch.lexicons import (
    load_role_lexicon,
    load_tech_synonyms,
    load_domain_lexicon,
)
from src.cvsearch.ingestion_pipeline import CVIngestionPipeline
from src.cvsearch.parser import parse_request, Criteria, TeamSize, TeamMember
from src.cvsearch.settings import Settings
from src.cvsearch.api_client import OpenAIClient


@click.group()
@click.pass_context
def cli(ctx):
    """RAG-Challenge-2 style CLI — step-by-step build."""
    settings = Settings()
    client = OpenAIClient(settings)
    db = CVDatabase(settings)
    ctx.obj = {"settings": settings, "client": client, "db": db}

@cli.command("env-info")
@click.pass_context
def env_info_cmd(ctx):
    """Print env detection (API key masked)."""
    settings: Settings = ctx.obj["settings"]
    def mask(s: str) -> str:
        if not s:
            return "(unset)"
        return (s[:4] + "..." + s[-4:]) if len(s) > 8 else "***"
    click.echo(f"--- Loaded from Settings ---")
    click.echo(f"OPENAI_API_KEY: {mask(settings.openai_api_key_str)}")
    click.echo(f"OPENAI_MODEL:   {settings.openai_model}")
    click.echo(f"SEARCH_MODE:    {settings.search_mode}")
    click.echo(f"DB_PATH:        {settings.db_path}")
    click.echo(f"LEXICON_DIR:    {settings.lexicon_dir}")

@cli.command("init-db")
@click.pass_context
def init_db_cmd(ctx):
    db: CVDatabase = ctx.obj["db"]
    try:
        db.initialize_schema()
        click.echo(f"Initialized database at {db.db_path}")
    finally:
        db.close()

@cli.command("check-db")
@click.pass_context
def check_db_cmd(ctx):
    db: CVDatabase = ctx.obj["db"]
    try:
        names = ", ".join(db.check_tables())
        click.echo(f"Tables: {names}")
        click.echo(f"FTS5: {db.check_fts()}")
    finally:
        db.close()

@cli.command("show-lexicons")
@click.pass_context
def show_lexicons_cmd(ctx):
    settings: Settings = ctx.obj["settings"]
    roles = load_role_lexicon(settings.lexicon_dir)
    techs = load_tech_synonyms(settings.lexicon_dir)
    doms = load_domain_lexicon(settings.lexicon_dir)
    click.echo(f"Roles: {len(roles)} | Tech groups: {len(techs)} | Domains: {len(doms)}")
    for k, v in list(techs.items())[:3]:
        click.echo(f"  {k}: {', '.join(v[:3])}{'...' if len(v)>3 else ''}")

@cli.command("ingest-mock")
@click.pass_context
def ingest_mock_cmd(ctx):
    settings: Settings = ctx.obj["settings"]
    db: CVDatabase = ctx.obj["db"]

    try:
        pipeline = CVIngestionPipeline(db, settings)
        n = pipeline.run_mock_ingestion()
        click.echo(f"Ingested {n} mock CVs into {db.db_path}")
        click.echo(f"Successfully built FAISS index at {settings.faiss_index_path}")
    finally:
        db.close()

@cli.command("parse-request")
@click.option("--text", type=str, required=True, help="Free-text client brief",
              default='For a new fintech project raising round we need to create a brand new code development .Field: healthcare.  Need  2 .net backend dev senior on .NET microservices, Kubernetes,k8s, PostgreSQL and possibly python codebase support, power bi , Kafka, react, ts, playwright.')
@click.option("--model", type=str, default=None, help="Override model, e.g. gpt-4.1-mini")
@click.pass_context
def parse_request_cmd(ctx, text, model):
    settings: Settings = ctx.obj["settings"]
    client: OpenAIClient = ctx.obj["client"]
    model_name = model or settings.openai_model
    c = parse_request(text, model=model_name, settings=settings, client=client)
    click.echo(c.to_json())

@cli.command("search-seat")
@click.option("--criteria", type=click.Path(exists=True, dir_okay=False), default='./criteria.json', required=True, help="Path to canonical criteria JSON")
@click.option("--topk", type=int, default=3, help="Top-K results to return")
@click.option("--mode", type=click.Choice(["lexical","semantic","hybrid"]), default=None, help="Ranking mode")
@click.option("--vs-topk", type=int, default=None, help="Vector-store fan-in (K)")
@click.option("--run-dir", type=click.Path(file_okay=False), default=None, help="Output folder for artifacts (default: runs/<timestamp>/)")
@click.option("--justify/--no-justify", default=True, help="Enable/disable LLM-based justification (slower).")
@click.pass_context
def search_seat_cmd(ctx, criteria, topk, mode, vs_topk, run_dir, justify):
    """
    Seat-aware search with strict gating, then lexical/semantic/hybrid ranking.
    """
    settings: Settings = ctx.obj["settings"]
    client: OpenAIClient = ctx.obj["client"]
    db: CVDatabase = ctx.obj["db"]

    try:
        with open(criteria, "r", encoding="utf-8") as f:
            crit = json.load(f)

        out_dir = run_dir or default_run_dir()

        processor = SearchProcessor(db, client, settings)
        payload = processor.search_for_seat(
            criteria=crit,
            top_k=topk,
            run_dir=out_dir,
            mode_override=mode,
            vs_topk_override=vs_topk,
            with_justification=justify
        )

        top_ids = [r["candidate_id"] for r in payload["results"]]
        click.echo(json.dumps({
            "run_dir": out_dir,
            "mode": mode or settings.search_mode,
            "topK": top_ids,
            "payload": payload
        }, indent=2, ensure_ascii=False))

    finally:
        db.close()

@cli.command("presale-plan")
@click.option("--text", type=str, required=True, help="Free-text client brief")
@click.pass_context
def presale_plan_cmd(ctx, text):
    """
    Stateless, budget-agnostic presale role composition strictly from the brief.
    """
    settings: Settings = ctx.obj["settings"]
    client: OpenAIClient = ctx.obj["client"]

    planner = Planner()
    crit = parse_request(text, model=settings.openai_model, settings=settings, client=client)
    plan = planner.derive_presale_team(crit, raw_text=text)

    click.echo(json.dumps(plan, indent=2, ensure_ascii=False))

@cli.command("project-search")
@click.option("--text", type=str, required=False, help="Free-text brief; used when --criteria is not provided")
@click.option("--criteria", type=click.Path(exists=True, dir_okay=False), required=False, help="Canonical criteria JSON (optional). If provided, seats are taken from here; otherwise derived from --text.")
@click.option("--topk", type=int, default=3, help="Top-K per seat")
@click.option("--run-dir", type=click.Path(file_okay=False), default=None, help="Base output folder for artifacts (default: runs/<timestamp>/)")
@click.option("--justify/--no-justify", default=True, help="Enable/disable LLM-based justification (slower).")
@click.pass_context
def project_search_cmd(ctx, text, criteria, topk, run_dir, justify):
    """
    Multi-seat search:
      - If --criteria is given, run per-seat search as-is.
      - Else, parse --text and derive seats deterministically, then search.
    """
    settings: Settings = ctx.obj["settings"]
    client: OpenAIClient = ctx.obj["client"]
    db: CVDatabase = ctx.obj["db"]

    if not criteria and not text:
        raise click.ClickException("Provide either --criteria or --text.")

    try:
        out_dir = run_dir or default_run_dir()

        processor = SearchProcessor(db, client, settings)

        if criteria:
            with open(criteria, "r", encoding="utf-8") as f:
                crit_dict = json.load(f)

            ts_dict = crit_dict.get("team_size", {})
            members = [
                TeamMember(
                    role=m["role"],
                    seniority=m.get("seniority"),
                    domains=m.get("domains", []),
                    tech_tags=m.get("tech_tags", []),
                    nice_to_have=m.get("nice_to_have", []),
                    rationale=m.get("rationale")
                ) for m in ts_dict.get("members", [])
            ]
            team_size_obj = TeamSize(
                total=ts_dict.get("total"),
                members=members
            )

            crit = Criteria(
                domain=crit_dict.get("domain", []),
                tech_stack=crit_dict.get("tech_stack", []),
                expert_roles=crit_dict.get("expert_roles", []),
                project_type=crit_dict.get("project_type"),
                team_size=team_size_obj
            )
            payload = processor.search_for_project(
                criteria=crit,
                top_k=topk,
                run_dir=out_dir,
                raw_text=None,
                with_justification=justify
            )
        else:
            crit = parse_request(text, model=settings.openai_model, settings=settings, client=client)
            payload = processor.search_for_project(
                criteria=crit,
                top_k=topk,
                run_dir=out_dir,
                raw_text=text,
                with_justification=justify
            )

        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))

    finally:
        db.close()


@cli.command("sync-gdrive")
@click.pass_context
def sync_gdrive_cmd(ctx):
    """
    Syncs files from a Google Drive folder to a local directory using rclone.

    This command requires rclone to be installed. You must first run
    'rclone config' to set up a Google Drive remote.

    Configure settings in your .env file:
    - GDRIVE_RCLONE_CONFIG_PATH (optional)
    - GDRIVE_REMOTE_NAME
    - GDRIVE_SOURCE_DIR
    - GDRIVE_LOCAL_DEST_DIR
    """
    settings: Settings = ctx.obj["settings"]
    db: CVDatabase = ctx.obj["db"]  # Get db to ensure we close it

    try:
        syncer = GDriveSyncer(settings)

        click.secho(
            f"Starting sync from GDrive remote '{settings.gdrive_remote_name}'...",
            fg="cyan"
        )

        syncer.sync_files()

        click.secho(
            f"\n✅ Google Drive sync completed successfully.",
            fg="green"
        )
        click.echo(
            f"Files are available in: {settings.gdrive_local_dest_dir}"
        )

    except FileNotFoundError as e:
        click.secho(f"\n❌ Error: {e}", fg="red")
        click.secho(
            "Please ensure 'rclone' is installed and in your system's PATH.",
            fg="red"
        )
        if settings.gdrive_rclone_config_path:
            click.secho(
                f"Also check that your config file exists at: "
                f"{settings.gdrive_rclone_config_path}",
                fg="red"
            )
        ctx.exit(1)

    except subprocess.CalledProcessError as e:
        click.secho(
            f"\n❌ rclone command failed with return code {e.returncode}.",
            fg="red"
        )
        click.secho(
            "Check the rclone output above for error details.",
            fg="red"
        )
        ctx.exit(1)

    except Exception as e:
        click.secho(f"\n❌ An unexpected error occurred: {e}", fg="red")
        ctx.exit(1)

    finally:
        db.close()

# --- START NEW HELPER FUNCTION ---
def _normalize_folder_name(name: str) -> str:
    """Converts a folder name to a potential lexicon key."""
    s = name.lower().strip()
    s = re.sub(r'\s+', '_', s) # Replace spaces with underscores
    s = re.sub(r'[^a-z0-9_]', '', s) # Remove non-alphanumeric chars
    return s
# --- END NEW HELPER FUNCTION ---


# --- START MODIFIED HELPER FUNCTION ---
def _process_single_cv_file(
        file_path: Path,
        parser: CVParser,
        client: OpenAIClient,
        settings: Settings,
        json_output_dir: Path,
        inbox_dir: Path,            # <-- NEW ARG
        role_keys_lookup: set[str]  # <-- NEW ARG
) -> tuple[str, dict | tuple[Path, str] | Path]:
    """
    Worker function to process one CV file.
    This is designed to be run in a ThreadPoolExecutor.

    Returns a tuple of (status, data):
    - ("processed", (Path, cv_dict))
    - ("skipped_role", (Path, missing_role_key))
    - ("skipped_ambiguous", Path)
    - ("failed_parsing", Path)
    """
    try:
        # --- 1. Find Role Hint and Apply Quality Gate ---
        relative_path = file_path.relative_to(inbox_dir)
        source_gdrive_path_str = str(relative_path.as_posix())

        path_parts = relative_path.parent.parts

        if not path_parts:
            # File is in the root of inbox_dir (e.g., gdrive_inbox/cv.pptx)
            # This is ambiguous, we don't know the category or role.
            return "skipped_ambiguous", file_path

        source_category = path_parts[0] # e.g., CANDIDATES or EMPLOYEES

        if len(path_parts) < 2:
            # File is in a category root (e.g., CANDIDATES/cv.pptx)
            # Also ambiguous, no role folder.
            return "skipped_ambiguous", file_path

        # This is the "Role-Gated Ingestion" logic
        role_folder_name = path_parts[1] # e.g., "Analytics Engineer", "Ruby", "Yaroslav Siomka"
        role_key = _normalize_folder_name(role_folder_name)

        if role_key not in role_keys_lookup:
            # THE GATE: Role key not in lexicon, skip this file.
            return "skipped_role", (file_path, role_key)

        # If we are here, the gate passed.
        source_folder_role_hint = role_key

        # --- 2. Process the file (slow part) ---
        click.echo(f"  -> Processing {file_path.name} (Role: {role_key})...")

        # Component 1: Extract text from PPTX
        raw_text = parser.extract_text(file_path)

        # Component 2: Map text to normalized JSON (The slow I/O part)
        cv_data_dict = client.get_structured_cv(
            raw_text, settings.openai_model, settings
        )

        # --- 3. Add All Metadata ---
        ingestion_time = datetime.now()

        file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
        cv_data_dict["candidate_id"] = f"pptx-{file_hash[:10]}"

        file_stat = file_path.stat()
        mod_time = datetime.fromtimestamp(file_stat.st_mtime)
        cv_data_dict["last_updated"] = mod_time.isoformat()

        # Add new fields
        cv_data_dict["source_filename"] = file_path.name
        cv_data_dict["ingestion_timestamp"] = ingestion_time.isoformat()
        cv_data_dict["source_gdrive_path"] = source_gdrive_path_str
        cv_data_dict["source_category"] = source_category
        cv_data_dict["source_folder_role_hint"] = source_folder_role_hint

        # --- 4. Save JSON for debugging ---
        json_filename = f"{cv_data_dict['candidate_id']}.json"
        json_save_path = json_output_dir / json_filename
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(cv_data_dict, f, indent=2, ensure_ascii=False)

        # Return path and success data
        return "processed", (file_path, cv_data_dict)

    except Exception as e:
        # Catch errors per-file so the batch can continue
        click.secho(f"  -> FAILED to parse {file_path.name}: {e}", fg="red")
        # Return path and failure data
        return "failed_parsing", file_path
# --- END MODIFIED HELPER FUNCTION ---


@cli.command("ingest-gdrive")
@click.pass_context
def ingest_gdrive_cmd(ctx):
    """
    Parses .pptx CVs from the GDrive inbox, saves to JSON for debug,
    and ingests them into the database and FAISS index.
    """
    # 1. Setup services
    settings: Settings = ctx.obj["settings"]
    client: OpenAIClient = ctx.obj["client"]
    db: CVDatabase = ctx.obj["db"]
    try:
        parser = CVParser()
    except NameError:
        click.secho("CVParser not found. Make sure 'src/cvsearch/cv_parser.py' exists.", fg="red")
        db.close()
        ctx.exit(1)

    # --- START NEW: Load Role Lexicon for Gating ---
    try:
        roles_lex = load_role_lexicon(settings.lexicon_dir)
        role_keys_lookup = set(roles_lex.keys())
        click.echo(f"Loaded {len(role_keys_lookup)} role keys from lexicon for gating.")
    except Exception as e:
        click.secho(f"❌ FAILED to load role lexicon: {e}", fg="red")
        click.echo("Cannot proceed without role lexicon for gating.")
        db.close()
        ctx.exit(1)
    # --- END NEW ---

    # 2. Define paths
    inbox_dir = settings.gdrive_local_dest_dir
    archive_dir = inbox_dir.parent / "gdrive_archive"
    json_output_dir = settings.data_dir / "ingested_cvs_json"
    archive_dir.mkdir(exist_ok=True)
    json_output_dir.mkdir(exist_ok=True)

    # 3. Find files to process (RECURSIVE)
    pptx_files = list(inbox_dir.rglob("*.pptx")) # <-- CHANGED to rglob

    # Filter out any files in an _archive folder (just in case)
    pptx_files = [p for p in pptx_files if "_archive" not in str(p.parent).lower()]

    if not pptx_files:
        click.echo(f"No .pptx files found in {inbox_dir}")
        db.close()
        return

    click.echo(f"Found {len(pptx_files)} .pptx CV(s) to process...")

    # 4. Loop, Extract, and Map (Parallelized)
    cvs_to_ingest = []
    processed_files = []
    failed_files = []
    skipped_ambiguous = []
    # Use defaultdict to aggregate skipped roles
    skipped_roles = defaultdict(list)

    max_workers = min(10, len(pptx_files))

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    _process_single_cv_file,
                    file_path,
                    parser,
                    client,
                    settings,
                    json_output_dir,
                    inbox_dir,          # <-- NEW ARG
                    role_keys_lookup    # <-- NEW ARG
                ): file_path for file_path in pptx_files
            }

            for future in as_completed(future_to_path):
                # Unpack the new return structure
                status, data = future.result()

                if status == "processed":
                    file_path, cv_data = data
                    cvs_to_ingest.append(cv_data)
                    processed_files.append(file_path)
                elif status == "failed_parsing":
                    failed_files.append(data)
                elif status == "skipped_ambiguous":
                    skipped_ambiguous.append(data)
                elif status == "skipped_role":
                    file_path, missing_role_key = data
                    skipped_roles[missing_role_key].append(file_path)

        # --- 5. Print Summary Report & Archive ---

        # 5a. Print the Data Quality Gate report (VERY IMPORTANT)
        if skipped_roles:
            click.secho("\n--- Data Quality Gate: Skipped CVs ---", fg="yellow")
            click.secho("The following CVs were skipped because their role folder is not in 'role_lexicon.json'.", fg="yellow")
            for role_key, files in skipped_roles.items():
                click.echo(f"  - Missing Role: '{role_key}' (Skipped {len(files)} CV(s))")
            click.secho("Please add these keys to the lexicon and re-run ingestion.", fg="yellow")

        if skipped_ambiguous:
            click.secho("\n--- Skipped Ambiguous CVs ---", fg="yellow")
            click.secho("The following CVs were skipped because they were not in a role folder:", fg="yellow")
            for file_path in skipped_ambiguous:
                click.echo(f"  - {file_path.relative_to(inbox_dir)}")

        # 5b. Archive *only* successfully processed files
        click.echo(f"\nArchiving {len(processed_files)} successfully processed file(s)...")
        for file_path in processed_files:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_filename = f"{file_path.stem}_archived_at_{timestamp}{file_path.suffix}"
            archive_dest_path = archive_dir / archive_filename
            try:
                shutil.move(str(file_path), str(archive_dest_path))
                click.secho(f"  -> Archived {file_path.name} as {archive_filename}", fg="green")
            except Exception as e:
                click.secho(f"  -> FAILED to archive {file_path.name}: {e}", fg="red")

        if failed_files:
            click.secho(f"{len(failed_files)} file(s) failed to parse and were not archived. See errors above.", fg="yellow")

        # 6. Ingest processed CVs into DB and FAISS
        if not cvs_to_ingest:
            click.echo("\nNo new CVs to ingest.")
            return

        click.echo(f"\nIngesting {len(cvs_to_ingest)} processed CV(s) into database...")
        pipeline = CVIngestionPipeline(db, settings)

        count = pipeline.upsert_cvs(cvs_to_ingest)

        click.secho(
            f"✅ Successfully upserted {count} new CV(s). Index is updated.",
            fg="green"
        )
        click.echo(f"Debug JSON files saved in: {json_output_dir}")

        unmapped = [
            cv.get("unmapped_tags") for cv in cvs_to_ingest
            if cv.get("unmapped_tags")
        ]
        if unmapped:
            click.secho("Review: The following tags were found but are not in your lexicons:", fg="yellow")
            all_unmapped = ", ".join(set(t.strip() for tags in unmapped for t in tags.split(',')))
            click.echo(all_unmapped)

    except Exception as e:
        click.secho(f"❌ FAILED during database ingestion: {e}", fg="red")
    finally:
        db.close()


if __name__ == "__main__":
    cli()