from __future__ import annotations

from typing import List, Dict, Any, Iterable, Tuple
import sqlite3, json, os
import faiss
import numpy as np
from src.cvsearch.data import load_mock_cvs
from src.cvsearch.settings import Settings
from src.cvsearch.api_client import OpenAIClient
from src.cvsearch.storage import CVDatabase
from src.cvsearch.local_embedder import LocalEmbedder

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

    def _build_global_faiss_index(self, embedding_data: List[Tuple[str, str]]) -> None:
        """
        Builds and saves a global FAISS index from the provided candidate text blobs.
        This follows the RAG-challenge's VectorDBIngestor pattern.
        """
        if not embedding_data:
            print("No candidate data provided to build FAISS index. Skipping.")
            return

        print(f"Starting FAISS index build for {len(embedding_data)} candidates...")

        self.db.clear_faiss_id_map()

        candidate_ids = [d[0] for d in embedding_data]
        texts_to_embed = [d[1] for d in embedding_data]

        mappings: List[Tuple[int, str]] = [(i, cid) for i, cid in enumerate(candidate_ids)]

        all_embeddings = self.local_embedder.get_embeddings(texts_to_embed)

        dims = self.local_embedder.dims
        index = faiss.IndexFlatIP(dims)

        embeddings_array = np.array(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        print(f"FAISS index created with {index.ntotal} vectors.")

        index_path = str(self.settings.faiss_index_path)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)

        self.db.insert_faiss_id_map_batch(mappings)
        self.db.commit()

        print(f"FAISS index saved to: {index_path}")
        print(f"FAISS ID map saved to SQLite table 'faiss_id_map'.")

    def run_ingestion_from_list(self, cvs: List[Dict[str, Any]]) -> int:
        """
        Ingest all CVs from a list, transactionally. Safe to re-run.
        """
        embedding_data: List[Tuple[str, str]] = []
        try:
            for cv in cvs:
                data_for_embedding = self._ingest_single_cv(cv)
                embedding_data.append(data_for_embedding)

            self.db.commit()
            print(f"Successfully ingested {len(cvs)} candidates into SQLite.")

        except Exception as e:
            print(f"Error during SQLite ingestion: {e}. Rolling back.")
            self.db.conn.rollback()
            raise

        try:
            self._build_global_faiss_index(embedding_data)
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise

        return len(cvs)

    def run_mock_ingestion(self) -> int:
        """
        Loads mock CVs and runs the ingestion pipeline.
        """
        cvs = load_mock_cvs(self.settings.data_dir)
        return self.run_ingestion_from_list(cvs)