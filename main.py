from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

# --- NEW IMPORTS for Component 3 ---
import hashlib
import shutil
from datetime import datetime
from src.cvsearch.cv_parser import CVParser
# --- END NEW IMPORTS ---

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

# --- START MODIFIED COMMAND ---

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

    # 2. Define paths
    inbox_dir = settings.gdrive_local_dest_dir
    archive_dir = inbox_dir / "_archive"
    json_output_dir = settings.data_dir / "ingested_cvs_json" # JSON output path
    archive_dir.mkdir(exist_ok=True)
    json_output_dir.mkdir(exist_ok=True) # Create JSON dir

    # 3. Find files to process
    pptx_files = list(inbox_dir.glob("*.pptx"))
    if not pptx_files:
        click.echo(f"No .pptx files found in {inbox_dir}")
        db.close()
        return

    click.echo(f"Found {len(pptx_files)} .pptx CV(s) to process...")
    cvs_to_ingest = []

    # 4. Loop, Extract, and Map
    #    The database connection remains open for the entire loop
    #    so that get_or_create_faiss_id works in a single transaction.
    try:
        for file_path in pptx_files:
            try:
                click.echo(f"  -> Processing {file_path.name}...")

                # Component 1: Extract text from PPTX
                raw_text = parser.extract_text(file_path)

                # Component 2: Map text to normalized JSON
                cv_data_dict = client.get_structured_cv(
                    raw_text, settings.openai_model, settings
                )

                # Generate and add missing metadata
                file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
                cv_data_dict["candidate_id"] = f"pptx-{file_hash[:10]}"
                cv_data_dict["last_updated"] = datetime.now().isoformat().split('T')[0]

                # Save JSON for debugging
                json_filename = f"{cv_data_dict['candidate_id']}.json"
                json_save_path = json_output_dir / json_filename
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(cv_data_dict, f, indent=2, ensure_ascii=False)

                cvs_to_ingest.append(cv_data_dict)

                # Move processed file to archive
                shutil.move(str(file_path), str(archive_dir / file_path.name))
                click.secho(f"  -> Successfully parsed, saved to JSON, and archived {file_path.name}", fg="green")

            except Exception as e:
                # Catch errors per-file so the batch can continue
                click.secho(f"  -> FAILED to parse {file_path.name}: {e}", fg="red")
                # You could move to a '_failed' directory here if desired

        # 5. Ingest processed CVs into DB and FAISS
        if not cvs_to_ingest:
            click.echo("No CVs were successfully processed.")
            return

        click.echo(f"Ingesting {len(cvs_to_ingest)} processed CV(s) into database...")
        pipeline = CVIngestionPipeline(db, settings)

        # --- THIS IS THE KEY CHANGE ---
        # Call the new upsert method instead of run_ingestion_from_list
        count = pipeline.upsert_cvs(cvs_to_ingest)
        # --- END KEY CHANGE ---

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
        # This will catch failures in the pipeline.upsert_cvs() call
        click.secho(f"❌ FAILED during database ingestion: {e}", fg="red")
    finally:
        # This ensures the DB connection is closed no matter what
        db.close()

# --- END MODIFIED COMMAND ---

if __name__ == "__main__":
    cli()