from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any # <-- NEW/MODIFIED IMPORTS

# --- REMOVED IMPORTS ---
# import re
# from collections import defaultdict
# import hashlib
# import shutil
# from datetime import datetime
# from src.cvsearch.cv_parser import CVParser
# from concurrent.futures import ThreadPoolExecutor, as_completed
# --- END REMOVED IMPORTS ---

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from src.cvsearch.search_processor import SearchProcessor, default_run_dir
from src.cvsearch.planner import Planner
import subprocess
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
@click.option("--text", type=str, required=False, help="Free-t'ext brief; used when --criteria is not provided")
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


def _print_gdrive_report(report: Dict[str, Any]):
    """Helper to print the ingestion report to the console."""

    if report.get("status") == "no_files_found":
        return # Message already printed by pipeline

    processed_count = report.get("processed_count", 0)
    skipped_roles = report.get("skipped_roles", {})
    skipped_ambiguous = report.get("skipped_ambiguous", [])
    failed_files = report.get("failed_files", [])
    archival_failures = report.get("archival_failures", [])
    unmapped_tags = report.get("unmapped_tags", [])
    json_output_dir = report.get("json_output_dir", "data/ingested_cvs_json")

    if skipped_roles:
        click.secho("\n--- Data Quality Gate: Skipped CVs ---", fg="yellow")
        click.secho("The following CVs were skipped because their role folder could not be mapped to a known role in 'role_lexicon.json'.", fg="yellow")
        for role_key, files in skipped_roles.items():
            click.echo(f"  - Unmapped Role Folder: '{role_key}' (Skipped {len(files)} CV(s))")
        click.secho("The LLM determined these are not valid role folders.", fg="yellow")

    if skipped_ambiguous:
        click.secho("\n--- Skipped Ambiguous CVs ---", fg="yellow")
        click.secho("The following CVs were skipped because they were not in a role folder:", fg="yellow")
        for file_path in skipped_ambiguous:
            click.echo(f"  - {file_path}")

    if archival_failures:
        click.secho("\n--- Archival Failures ---", fg="red")
        for file_name, error_msg in archival_failures:
            click.echo(f"  - FAILED to archive {file_name}: {error_msg}")

    if failed_files:
        click.secho(f"\n{len(failed_files)} file(s) failed to parse and were not archived. See errors above.", fg="red")

    if unmapped_tags:
        click.secho("\n--- Lexicon Review ---", fg="yellow")
        click.secho("The following tags were found but are not in your lexicons:", fg="yellow")
        click.echo(", ".join(unmapped_tags))

    click.echo(f"\nDebug JSON files saved in: {json_output_dir}")
    click.secho(f"\n✅ GDrive Ingestion Complete: {processed_count} CV(s) upserted.", fg="green")


@cli.command("ingest-gdrive")
@click.pass_context
def ingest_gdrive_cmd(ctx):
    """
    Parses .pptx CVs from the GDrive inbox, saves to JSON for debug,
    and ingests them into the database and FAISS index.
    """
    settings: Settings = ctx.obj["settings"]
    client: OpenAIClient = ctx.obj["client"]
    db: CVDatabase = ctx.obj["db"]

    try:
        pipeline = CVIngestionPipeline(db, settings)
        report = pipeline.run_gdrive_ingestion(client)
        _print_gdrive_report(report)

    except Exception as e:
        click.secho(f"❌ FAILED during database ingestion: {e}", fg="red")
    finally:
        db.close()


if __name__ == "__main__":
    cli()