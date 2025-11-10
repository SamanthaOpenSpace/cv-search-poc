from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import click
import faiss
import numpy as np

from cv_search.clients.openai_client import OpenAIClient
from cv_search.config.settings import Settings
from cv_search.db.database import CVDatabase
from cv_search.ingestion.cv_parser import CVParser
from cv_search.ingestion.data_loader import load_mock_cvs
from cv_search.lexicon.loader import load_role_lexicon
from cv_search.retrieval.local_embedder import LocalEmbedder

class CVIngestionPipeline:
    """
    Orchestrates the complete ingestion of CV data into the database
    and (now) a local FAISS index.
    """
    def __init__(self, db: CVDatabase, settings: Settings):
        self.db = db
        self.settings = settings
        self.local_embedder = LocalEmbedder()

    def _canon_tags(self, seq: Iterable[str]) -> List[str]:
        return self._uniq([(s or "").strip().lower() for s in (seq or [])])

    def _uniq(self, seq: Iterable[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in seq:
            xl = (x or "").strip()
            if not xl:
                continue
            k = xl.lower()
            if k not in seen:
                seen.add(k)
                out.append(xl)
        return out

    def _mk_experience_line(self, exp: Dict[str, Any]) -> str:
        title = exp.get("title", "")
        company = exp.get("company", "")
        domains = ", ".join(self._canon_tags(exp.get("domain_tags", []) or []))
        techs = ", ".join(self._canon_tags(exp.get("tech_tags", []) or []))
        highlights = " ; ".join(exp.get("highlights", []) or [])
        return (
            f"{title} @ {company}"
            f"{' | domains: ' + domains if domains else ''}"
            f"{' | tech: ' + techs if techs else ''}"
            f"{' | highlights: ' + highlights if highlights else ''}"
        )

    def _build_candidate_doc_texts(self, cv: Dict[str, Any], domain_rollup: List[str]) -> Tuple[str, str, str]:
        summary_text = cv.get("summary", "") or ""
        exp_lines = [self._mk_experience_line(e) for e in (cv.get("experience", []) or [])]
        experience_text = " \n".join([ln for ln in exp_lines if ln])
        roles   = self._canon_tags(cv.get("role_tags", []) or [])
        techs   = self._canon_tags(cv.get("tech_tags", []) or [])
        senior  = self._canon_tags([cv.get("seniority", "")]) if cv.get("seniority") else []
        domains = self._canon_tags(domain_rollup)
        distinct = self._uniq(roles + techs + domains + senior)
        tags_text = " ".join(distinct)
        return summary_text, experience_text, tags_text

    def _ingest_single_cv(self, cv: Dict[str, Any]) -> Tuple[str, str]:
        """
        Ingest a single CV document into SQLite and return its text
        blob for embedding.
        """
        candidate_id = cv["candidate_id"]

        self.db.upsert_candidate(cv)

        self.db.remove_candidate_derived(candidate_id)

        role_tags_top = self._canon_tags(cv.get("role_tags", []) or [])
        tech_tags_top = self._canon_tags(cv.get("tech_tags", []) or [])
        seniority = (cv.get("seniority", "") or "").strip().lower()

        experiences = cv.get("experience", []) or []
        domain_tags_list = [self._canon_tags(exp.get("domain_tags", []) or []) for exp in experiences]
        tech_tags_list = [self._canon_tags(exp.get("tech_tags", []) or []) for exp in experiences]
        domain_rollup = self._canon_tags([tag for sublist in domain_tags_list for tag in sublist])

        self.db.insert_experiences_and_tags(
            candidate_id,
            experiences,
            domain_tags_list,
            tech_tags_list
        )

        self.db.upsert_candidate_tags(
            candidate_id,
            role_tags=role_tags_top,
            tech_tags_top=tech_tags_top,
            seniority=seniority,
            domain_rollup=domain_rollup,
        )

        summary_text, experience_text, tags_text = self._build_candidate_doc_texts(cv, domain_rollup)
        self.db.upsert_candidate_doc(
            candidate_id,
            summary_text,
            experience_text,
            tags_text,
            last_updated=cv.get("last_updated", "") or "",
            location=cv.get("location", "") or "",
            seniority=seniority,
        )

        vs_attributes = {
            "candidate_id": candidate_id,
            "role": role_tags_top[0] if role_tags_top else "",
            "seniority": seniority,
            "domains": domain_rollup,
            "tech": tech_tags_top,
        }

        role = (vs_attributes.get("role") or "") if isinstance(vs_attributes.get("role"), str) else (vs_attributes.get("role") or "")
        header = (
            f"candidate_id={candidate_id}"
            f" | role={role}"
            f" | seniority={vs_attributes.get('seniority') or ''}"
            f" | domains=[{', '.join(vs_attributes.get('domains') or [])}]"
            f" | tech=[{', '.join(vs_attributes.get('tech') or [])}]"
        )
        parts = [
            header.strip(),
            "",
            (tags_text or "").strip(),
            "---",
            (summary_text or "").strip(),
            "---",
            (experience_text or "").strip(),
        ]
        vs_text = "\n".join(parts).strip() + "\n"

        return (candidate_id, vs_text)

    # --- START NEW METHODS ---

    def load_or_create_index(self) -> faiss.IndexIDMap:
        """
        Safely loads the FAISS index from disk.
        If it doesn't exist, creates a new IndexIDMap.
        """
        index_path = str(self.settings.faiss_index_path)
        if os.path.exists(index_path):
            try:
                print(f"Loading existing FAISS index from: {index_path}")
                index = faiss.read_index(index_path)
                # Ensure it's an IndexIDMap (or can be cast to one)
                if not isinstance(index, faiss.IndexIDMap):
                    print(f"Warning: Index is not an IndexIDMap. Re-creating.")
                    # This is a recovery step if the old index type is found
                    return self._create_new_index()
                print(f"FAISS index loaded. Total vectors: {index.ntotal}")
                return index
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Re-creating.")
                return self._create_new_index()
        else:
            return self._create_new_index()

    def _create_new_index(self) -> faiss.IndexIDMap:
        """Helper to create a new, empty IndexIDMap."""
        print(f"No FAISS index found. Creating new IndexIDMap...")
        dims = self.local_embedder.dims
        # Create the core flat index (IP = Inner Product for cosine similarity)
        index_flat = faiss.IndexFlatIP(dims)
        # Wrap it with IndexIDMap to allow 64-bit int IDs
        index = faiss.IndexIDMap(index_flat)
        print(f"New FAISS index created (dims: {dims}).")
        return index

    def upsert_cvs(self, cvs: List[Dict[str, Any]]) -> int:
        """
        Ingests a list of CVs into SQLite and "upserts" their
        vectors into the FAISS index using stable IDs.
        Manages its own database transaction.
        """
        if not cvs:
            print("No CVs provided for upsert. Skipping.")
            return 0

        print(f"Starting upsert process for {len(cvs)} CV(s)...")
        index = self.load_or_create_index()

        embeddings_to_add = []
        faiss_ids_to_add = []

        try:
            for cv in cvs:
                # 1. Ingest text/metadata into SQLite
                (candidate_id, vs_text) = self._ingest_single_cv(cv)

                # 2. Get stable, persistent 64-bit ID (from Phase 1)
                faiss_id = self.db.get_or_create_faiss_id(candidate_id)

                # 3. Get embedding for the candidate's text blob
                embedding = self.local_embedder.get_embeddings([vs_text])[0]

                embeddings_to_add.append(embedding)
                faiss_ids_to_add.append(faiss_id)

            print(f"Batching {len(embeddings_to_add)} vectors into FAISS index...")

            # 4. Batch add/overwrite vectors in FAISS
            embeddings_array = np.array(embeddings_to_add).astype('float32')
            ids_array = np.array(faiss_ids_to_add).astype('int64')

            faiss.normalize_L2(embeddings_array)
            # add_with_ids handles both new additions and updates
            index.add_with_ids(embeddings_array, ids_array)

            # 5. Save the updated index back to disk
            index_path = str(self.settings.faiss_index_path)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)

            # 6. Commit all DB changes (candidate data, faiss_id_map)
            self.db.commit()

            print(f"FAISS index upsert complete. Total vectors: {index.ntotal}")
            print(f"Index saved to: {index_path}")
            print(f"Database changes for {len(cvs)} CVs committed.")

            return len(cvs)

        except Exception as e:
            print(f"Error during FAISS/DB upsert: {e}. Rolling back.")
            self.db.conn.rollback()
            raise

    # --- END NEW METHODS ---

    # --- DELETED METHODS ---
    # _build_global_faiss_index(self, ...)
    # run_ingestion_from_list(self, ...)
    # --- END DELETED METHODS ---

    def run_mock_ingestion(self) -> int:
        """
        Clears the database and FAISS index, then ingests mock CVs.
        This is a true "re-ingest" for development.
        """
        print("--- Running Mock Ingestion (Full Rebuild) ---")

        # 1. Delete old DB and FAISS files
        db_path = str(self.settings.db_path)
        index_path = str(self.settings.faiss_index_path)

        if os.path.exists(db_path):
            print(f"Removing old database: {db_path}")
            try:
                self.db.close() # Close connection before deleting
            except Exception as e:
                print(f"Could not close DB connection (may be closed): {e}")
            os.remove(db_path)

        if os.path.exists(index_path):
            print(f"Removing old FAISS index: {index_path}")
            os.remove(index_path)

        # 2. Re-initialize DB and schema
        # We need to create a new connection object for the new file
        self.db = CVDatabase(self.settings)
        print("Initializing new database schema...")
        self.db.initialize_schema() # This method auto-commits

        # 3. Load mock CVs (now from data/test)
        cvs = load_mock_cvs(self.settings.test_data_dir)
        print(f"Loaded {len(cvs)} mock CVs from JSON.")

        # 4. Call the new upsert method
        # This method now manages its own transaction
        count = self.upsert_cvs(cvs)

        print(f"--- Mock Ingestion Complete: {count} CVs ---")
        return count


    # --- START REFACTORED GDRIVE INGESTION LOGIC ---

    def _normalize_folder_name(self, name: str) -> str:
        """Converts a folder name to a potential lexicon key."""
        s = name.lower().strip()
        s = re.sub(r'\s+', '_', s) # Replace spaces with underscores
        s = re.sub(r'[^a-z0-9_]', '', s) # Remove non-alphanumeric chars
        return s

    def _process_single_cv_file(
            self,
            file_path: Path,
            parser: CVParser,
            client: OpenAIClient,
            json_output_dir: Path,
            inbox_dir: Path
    ) -> tuple[str, dict | tuple[Path, str] | Path]:
        """
        Worker function to process one CV file.
        This is designed to be run in a ThreadPoolExecutor.

        Returns a tuple of (status, data):
        - ("processed", (Path, cv_dict))
        - ("skipped_ambiguous", Path)
        - ("failed_parsing", Path)
        """
        try:
            # 1. Find Role Hint and Apply Quality Gate
            relative_path = file_path.relative_to(inbox_dir)
            source_gdrive_path_str = str(relative_path.as_posix())

            path_parts = relative_path.parent.parts

            if not path_parts:
                return "skipped_ambiguous", file_path

            source_category = path_parts[0]

            if len(path_parts) < 2:
                return "skipped_ambiguous", file_path

            role_folder_name = path_parts[1]
            role_key = self._normalize_folder_name(role_folder_name)

            # 2. Process the file (slow part)
            click.echo(f"  -> Processing {file_path.name} (Hint: {role_key})...")

            raw_text = parser.extract_text(file_path)

            cv_data_dict = client.get_structured_cv(
                raw_text,
                role_key,
                self.settings.openai_model,
                self.settings
            )

            # 3. Add All Metadata
            ingestion_time = datetime.now()

            file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
            cv_data_dict["candidate_id"] = f"pptx-{file_hash[:10]}"

            file_stat = file_path.stat()
            mod_time = datetime.fromtimestamp(file_stat.st_mtime)
            cv_data_dict["last_updated"] = mod_time.isoformat()

            cv_data_dict["source_filename"] = file_path.name
            cv_data_dict["ingestion_timestamp"] = ingestion_time.isoformat()
            cv_data_dict["source_gdrive_path"] = source_gdrive_path_str
            cv_data_dict["source_category"] = source_category

            # 4. Save JSON for debugging
            json_filename = f"{cv_data_dict['candidate_id']}.json"
            json_save_path = json_output_dir / json_filename
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(cv_data_dict, f, indent=2, ensure_ascii=False)

            return "processed", (file_path, cv_data_dict)

        except Exception as e:
            click.secho(f"  -> FAILED to parse {file_path.name}: {e}", fg="red")
            return "failed_parsing", file_path

    def run_gdrive_ingestion(self, client: OpenAIClient) -> Dict[str, Any]:
        """
        Parses .pptx CVs from the GDrive inbox, saves to JSON for debug,
        and ingests them into the database and FAISS index.
        Returns a report dictionary of the outcomes.
        """
        parser = CVParser()

        # Load role lexicon (used for logging skips, not hard gating)
        try:
            roles_lex = load_role_lexicon(self.settings.lexicon_dir)
            role_keys_lookup = set(roles_lex.keys())
            click.echo(f"Loaded {len(role_keys_lookup)} role keys from lexicon.")
        except Exception as e:
            click.secho(f"❌ FAILED to load role lexicon: {e}", fg="red")
            raise

        # Define paths
        inbox_dir = self.settings.gdrive_local_dest_dir
        archive_dir = inbox_dir.parent / "gdrive_archive"
        json_output_dir = self.settings.data_dir / "ingested_cvs_json"
        archive_dir.mkdir(exist_ok=True)
        json_output_dir.mkdir(exist_ok=True)

        # Find files to process (RECURSIVE)
        pptx_files = list(inbox_dir.rglob("*.pptx"))
        pptx_files = [p for p in pptx_files if "_archive" not in str(p.parent).lower()]

        if not pptx_files:
            click.echo(f"No .pptx files found in {inbox_dir}")
            return {"processed_count": 0, "status": "no_files_found"}

        click.echo(f"Found {len(pptx_files)} .pptx CV(s) to process...")

        # Loop, Extract, and Map (Parallelized)
        cvs_to_ingest = []
        processed_files = []
        failed_files = []
        skipped_ambiguous = []
        skipped_roles = defaultdict(list)

        max_workers = min(10, len(pptx_files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self._process_single_cv_file,
                    file_path,
                    parser,
                    client,
                    json_output_dir,
                    inbox_dir
                ): file_path for file_path in pptx_files
            }

            for future in as_completed(future_to_path):
                status, data = future.result()

                if status == "processed":
                    file_path, cv_data = data
                    if cv_data.get("source_folder_role_hint") is None:
                        role_key = self._normalize_folder_name(file_path.relative_to(inbox_dir).parent.parts[1])
                        skipped_roles[role_key].append(str(file_path))
                    else:
                        cvs_to_ingest.append(cv_data)
                        processed_files.append(file_path)
                elif status == "failed_parsing":
                    failed_files.append(str(data))
                elif status == "skipped_ambiguous":
                    skipped_ambiguous.append(str(data.relative_to(inbox_dir)))

        # Archive *only* successfully processed files
        archived_files = []
        archival_failures = []
        for file_path in processed_files:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_filename = f"{file_path.stem}_archived_at_{timestamp}{file_path.suffix}"
            archive_dest_path = archive_dir / archive_filename
            try:
                shutil.move(str(file_path), str(archive_dest_path))
                archived_files.append(archive_filename)
            except Exception as e:
                archival_failures.append((file_path.name, str(e)))

        # Ingest processed CVs into DB and FAISS
        ingested_count = 0
        if cvs_to_ingest:
            click.echo(f"\nIngesting {len(cvs_to_ingest)} processed CV(s) into database...")
            ingested_count = self.upsert_cvs(cvs_to_ingest)
            click.secho(f"✅ Successfully upserted {ingested_count} new CV(s). Index is updated.", fg="green")
        else:
            click.echo("\nNo new CVs to ingest.")

        unmapped = [
            cv.get("unmapped_tags") for cv in cvs_to_ingest
            if cv.get("unmapped_tags")
        ]
        all_unmapped_tags = []
        if unmapped:
            all_unmapped_tags = sorted(list(set(
                t.strip() for tags in unmapped for t in tags.split(',') if t.strip()
            )))

        return {
            "processed_count": ingested_count,
            "archived_files": archived_files,
            "archival_failures": archival_failures,
            "skipped_roles": dict(skipped_roles),
            "skipped_ambiguous": skipped_ambiguous,
            "failed_files": failed_files,
            "unmapped_tags": all_unmapped_tags,
            "json_output_dir": str(json_output_dir)
        }

    # --- END REFACTORED GDRIVE INGESTION LOGIC ---