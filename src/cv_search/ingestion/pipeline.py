from __future__ import annotations

import hashlib
import json
import os
import re
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

    def load_or_create_index(self) -> faiss.IndexIDMap:
        index_path = str(self.settings.faiss_index_path)
        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
                if not isinstance(index, faiss.IndexIDMap):
                    return self._create_new_index()
                return index
            except Exception:
                return self._create_new_index()
        else:
            return self._create_new_index()

    def _create_new_index(self) -> faiss.IndexIDMap:
        dims = self.local_embedder.dims
        index_flat = faiss.IndexFlatIP(dims)
        index = faiss.IndexIDMap(index_flat)
        return index

    def upsert_cvs(self, cvs: List[Dict[str, Any]]) -> int:
        if not cvs:
            return 0

        index = self.load_or_create_index()

        embeddings_to_add = []
        faiss_ids_to_add = []

        try:
            for cv in cvs:
                (candidate_id, vs_text) = self._ingest_single_cv(cv)
                faiss_id = self.db.get_or_create_faiss_id(candidate_id)
                embedding = self.local_embedder.get_embeddings([vs_text])[0]
                embeddings_to_add.append(embedding)
                faiss_ids_to_add.append(faiss_id)

            embeddings_array = np.array(embeddings_to_add).astype('float32')
            ids_array = np.array(faiss_ids_to_add).astype('int64')

            faiss.normalize_L2(embeddings_array)
            index.add_with_ids(embeddings_array, ids_array)

            index_path = str(self.settings.faiss_index_path)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)

            self.db.commit()

            return len(cvs)
        except Exception:
            self.db.conn.rollback()
            raise

    def run_mock_ingestion(self) -> int:
        db_path = str(self.settings.db_path)
        index_path = str(self.settings.faiss_index_path)

        if os.path.exists(db_path):
            try:
                self.db.close()
            except Exception:
                pass
            os.remove(db_path)

        if os.path.exists(index_path):
            os.remove(index_path)

        self.db = CVDatabase(self.settings)
        self.db.initialize_schema()

        cvs = load_mock_cvs(self.settings.test_data_dir)

        count = self.upsert_cvs(cvs)

        return count

    def _normalize_folder_name(self, name: str) -> str:
        s = name.lower().strip()
        s = re.sub(r'\s+', '_', s)
        s = re.sub(r'[^a-z0-9_]', '', s)
        return s

    def _process_single_cv_file(
            self,
            file_path: Path,
            parser: CVParser,
            client: OpenAIClient,
            json_output_dir: Path,
            inbox_dir: Path
    ) -> tuple[str, dict | tuple[Path, str] | Path]:
        try:
            relative_path = file_path.relative_to(inbox_dir)
            source_gdrive_path_str = str(relative_path.as_posix())

            path_parts = relative_path.parent.parts
            source_category = path_parts[0] if path_parts else None

            role_key = ""
            if len(path_parts) >= 2:
                role_key = self._normalize_folder_name(path_parts[1])

            hint_display = role_key if role_key else "n/a"
            click.echo(f"  -> Processing {file_path.name} (Hint: {hint_display})...")

            raw_text = parser.extract_text(file_path)

            cv_data_dict = client.get_structured_cv(
                raw_text,
                role_key,
                self.settings.openai_model,
                self.settings
            )

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

            json_filename = f"{cv_data_dict['candidate_id']}.json"
            json_save_path = json_output_dir / json_filename
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(cv_data_dict, f, indent=2, ensure_ascii=False)

            return "processed", (file_path, cv_data_dict)

        except Exception as e:
            click.secho(f"  -> FAILED to parse {file_path.name}: {e}", fg="red")
            return "failed_parsing", file_path

    def run_gdrive_ingestion(self, client: OpenAIClient, target_filename: str | None = None) -> Dict[str, Any]:
        parser = CVParser()

        try:
            roles_lex_list = load_role_lexicon(self.settings.lexicon_dir)
            role_keys_lookup = set(roles_lex_list)
            click.echo(f"Loaded {len(role_keys_lookup)} role keys from lexicon.")
        except Exception as e:
            click.secho(f"❌ FAILED to load role lexicon: {e}", fg="red")
            raise

        inbox_dir = self.settings.gdrive_local_dest_dir
        json_output_dir = self.settings.data_dir / "ingested_cvs_json"
        json_output_dir.mkdir(exist_ok=True)

        pptx_files = list(inbox_dir.rglob("*.pptx"))

        if target_filename:
            pptx_files = [p for p in pptx_files if p.name == target_filename]

        if not pptx_files:
            click.echo(f"No .pptx files found in {inbox_dir}")
            return {"processed_count": 0, "status": "no_files_found"}

        skipped_unchanged: List[str] = []
        filtered: List[Path] = []
        for p in pptx_files:
            mtime_iso = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            last_upd = self.db.get_candidate_last_updated_by_source_filename(p.name)
            if last_upd and last_upd == mtime_iso:
                skipped_unchanged.append(str(p.relative_to(inbox_dir)))
            else:
                filtered.append(p)

        if not filtered and skipped_unchanged:
            click.echo("No new or modified .pptx files to process.")
            return {
                "processed_count": 0,
                "status": "no_changes",
                "skipped_unchanged": skipped_unchanged,
                "skipped_roles": {},
                "skipped_ambiguous": [],
                "failed_files": [],
                "unmapped_tags": [],
                "json_output_dir": str(json_output_dir),
            }

        click.echo(f"Found {len(filtered)} .pptx CV(s) to process...")

        cvs_to_ingest = []
        processed_files = []
        failed_files = []
        skipped_ambiguous = []
        skipped_roles = defaultdict(list)

        max_workers = min(10, len(filtered))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(
                    self._process_single_cv_file,
                    file_path,
                    parser,
                    client,
                    json_output_dir,
                    inbox_dir
                ): file_path for file_path in filtered
            }

            for future in as_completed(future_to_path):
                status, data = future.result()

                if status == "processed":
                    file_path, cv_data = data
                    cvs_to_ingest.append(cv_data)
                    processed_files.append(file_path)
                elif status == "failed_parsing":
                    failed_files.append(str(data))
                elif status == "skipped_ambiguous":
                    skipped_ambiguous.append(str(data.relative_to(inbox_dir)))

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
            "skipped_roles": {},
            "skipped_ambiguous": skipped_ambiguous,
            "failed_files": failed_files,
            "unmapped_tags": all_unmapped_tags,
            "json_output_dir": str(json_output_dir),
            "skipped_unchanged": skipped_unchanged,
        }

    def run_ingestion_from_list(self, cvs: List[Dict[str, Any]]) -> int:
        return self.upsert_cvs(cvs)
