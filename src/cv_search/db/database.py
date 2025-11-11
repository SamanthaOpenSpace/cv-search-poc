from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cv_search.config.settings import Settings


class CVDatabase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = str(settings.db_path)
        self.schema_file = str(settings.schema_file)
        self.conn = self._get_db()

    def _get_db(self) -> sqlite3.Connection:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def initialize_schema(self) -> None:
        with open(self.schema_file, "r", encoding="utf-8") as handle:
            self.conn.executescript(handle.read())
        self.commit()

    def check_tables(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        return [r[0] for r in rows]

    def check_fts(self) -> str:
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts_test USING fts5(x);")
            self.conn.execute("DROP TABLE IF EXISTS __fts_test;")
            return "available"
        except sqlite3.OperationalError as exc:
            return f"not available: {exc}"

    def set_app_config(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO app_config(key, value)
            VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def get_app_config(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM app_config WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row else None

    def remove_candidate_derived(self, candidate_id: str) -> None:
        self.conn.execute("DELETE FROM faiss_id_map WHERE candidate_id = ?", (candidate_id,))
        self.conn.execute("DELETE FROM candidate_doc WHERE candidate_id = ?", (candidate_id,))
        exp_ids = [
            r[0]
            for r in self.conn.execute(
                "SELECT id FROM experience WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchall()
        ]
        if exp_ids:
            self.conn.executemany(
                "DELETE FROM experience_tag WHERE experience_id = ?",
                [(i,) for i in exp_ids],
            )
        self.conn.execute("DELETE FROM experience WHERE candidate_id = ?", (candidate_id,))
        self.conn.execute("DELETE FROM candidate_tag WHERE candidate_id = ?", (candidate_id,))

    def upsert_candidate(self, cv: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO candidate(
                candidate_id,
                name,
                location,
                seniority,
                last_updated,
                source_filename,
                source_gdrive_path,
                source_category,
                source_folder_role_hint
            )
            VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                name=excluded.name,
                                                 location=excluded.location,
                                                 seniority=excluded.seniority,
                                                 last_updated=excluded.last_updated,
                                                 source_filename=excluded.source_filename,
                                                 source_gdrive_path=excluded.source_gdrive_path,
                                                 source_category=excluded.source_category,
                                                 source_folder_role_hint=excluded.source_folder_role_hint
            """,
            (
                cv["candidate_id"],
                cv.get("name", "[redacted]"),
                cv.get("location", ""),
                cv.get("seniority", ""),
                cv.get("last_updated", ""),
                cv.get("source_filename", None),
                cv.get("source_gdrive_path", None),
                cv.get("source_category", None),
                cv.get("source_folder_role_hint", None),
            ),
        )

    def insert_experiences_and_tags(
            self,
            candidate_id: str,
            experiences: List[Dict[str, Any]],
            domain_tags_list: List[List[str]],
            tech_tags_list: List[List[str]],
    ) -> List[int]:
        exp_ids: List[int] = []
        exp_tech_tags_to_insert: List[Tuple[int, str, str]] = []
        exp_domain_tags_to_insert: List[Tuple[int, str, str]] = []

        for idx, exp in enumerate(experiences or []):
            domain_tags = domain_tags_list[idx]
            tech_tags = tech_tags_list[idx]

            cur = self.conn.execute(
                """
                INSERT INTO experience(
                    candidate_id,
                    title,
                    company,
                    start,
                    end,
                    domain_tags_csv,
                    tech_tags_csv,
                    highlights
                )
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    candidate_id,
                    exp.get("title", ""),
                    exp.get("company", ""),
                    exp.get("from", ""),
                    exp.get("to", ""),
                    ",".join(domain_tags),
                    ",".join(tech_tags),
                    "\n".join(exp.get("highlights", []) or []),
                ),
            )
            exp_id = int(cur.lastrowid)
            exp_ids.append(exp_id)

            for tag in tech_tags:
                exp_tech_tags_to_insert.append((exp_id, "tech", tag))
            for tag in domain_tags:
                exp_domain_tags_to_insert.append((exp_id, "domain", tag))

        self.conn.executemany(
            "INSERT INTO experience_tag(experience_id, tag_type, tag_key) VALUES (?,?,?) ON CONFLICT(experience_id, tag_type, tag_key) DO NOTHING",
            exp_tech_tags_to_insert,
        )
        self.conn.executemany(
            "INSERT INTO experience_tag(experience_id, tag_type, tag_key) VALUES (?,?,?) ON CONFLICT(experience_id, tag_type, tag_key) DO NOTHING",
            exp_domain_tags_to_insert,
        )

        return exp_ids

    def upsert_candidate_tags(
            self,
            candidate_id: str,
            role_tags: List[str],
            tech_tags_top: List[str],
            seniority: str,
            domain_rollup: List[str],
    ) -> None:
        tags_to_insert: List[Tuple[str, str, str, float]] = []

        for role in role_tags:
            if role:
                tags_to_insert.append((candidate_id, "role", role, 2.0))
        for tag in tech_tags_top:
            if tag:
                tags_to_insert.append((candidate_id, "tech", tag, 1.5))
        if seniority:
            tags_to_insert.append((candidate_id, "seniority", seniority, 1.0))
        for domain in domain_rollup:
            if domain:
                tags_to_insert.append((candidate_id, "domain", domain, 1.0))

        self.conn.executemany(
            """
            INSERT INTO candidate_tag(candidate_id, tag_type, tag_key, weight)
            VALUES (?,?,?,?)
                ON CONFLICT(candidate_id, tag_type, tag_key)
                DO UPDATE SET weight = excluded.weight
            """,
            tags_to_insert,
        )

    def upsert_candidate_doc(
            self,
            candidate_id: str,
            summary_text: str,
            experience_text: str,
            tags_text: str,
            last_updated: str,
            location: str,
            seniority: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO candidate_doc(candidate_id, summary_text, experience_text, tags_text, last_updated, location, seniority)
            VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                                                 experience_text = excluded.experience_text,
                                                 tags_text = excluded.tags_text,
                                                 last_updated = excluded.last_updated,
                                                 location = excluded.location,
                                                 seniority = excluded.seniority
            """,
            (
                candidate_id,
                summary_text,
                experience_text,
                tags_text,
                last_updated,
                location,
                seniority,
            ),
        )

    def map_candidate_to_faiss_id(self, candidate_id: str, faiss_id: int) -> None:
        self.conn.execute(
            """
            INSERT INTO faiss_id_map(candidate_id, faiss_id)
            VALUES (?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                faiss_id = excluded.faiss_id
            """,
            (candidate_id, faiss_id),
        )

    def get_candidate_ids_from_faiss_ids(self, faiss_ids: Iterable[int]) -> Dict[int, str]:
        rows = self.conn.execute(
            "SELECT candidate_id, faiss_id FROM faiss_id_map WHERE faiss_id IN ({})".format(
                ",".join(["?"] * len(tuple(faiss_ids))) if faiss_ids else "NULL"
            ),
            tuple(faiss_ids),
        ).fetchall()
        return {row["faiss_id"]: row["candidate_id"] for row in rows}

    def fetch_tag_hits(self, candidate_ids: List[str], tags: List[str]) -> Dict[str, Dict[str, bool]]:
        if not candidate_ids or not tags:
            return {}
        rows = self.conn.execute(
            """
            SELECT candidate_id, tag_key
            FROM candidate_tag
            WHERE candidate_id IN ({})
              AND tag_key IN ({})
            """.format(
                ",".join(["?"] * len(candidate_ids)),
                ",".join(["?"] * len(tags)),
            ),
            tuple(candidate_ids) + tuple(tags),
            ).fetchall()
        result: Dict[str, Dict[str, bool]] = {}
        for row in rows:
            result.setdefault(row["candidate_id"], {})[row["tag_key"]] = True
        return result

    def compute_idf(self, tokens: List[str], tag_type: str) -> Dict[str, float]:
        if not tokens:
            return {}
        rows = self.conn.execute(
            """
            SELECT tag_key, COUNT(*) AS df
            FROM candidate_tag
            WHERE tag_type = ?
              AND tag_key IN ({})
            GROUP BY tag_key
            """.format(
                ",".join(["?"] * len(tokens)),
            ),
            (tag_type, *tokens),
        ).fetchall()
        total_candidates = self.conn.execute("SELECT COUNT(*) FROM candidate").fetchone()[0]
        idf: Dict[str, float] = {}
        for row in rows:
            df = row["df"]
            idf[row["tag_key"]] = math.log((total_candidates + 1) / (df + 1)) + 1
        return idf

    def rank_weighted_set(
            self,
            gated_ids: List[str],
            must_have: List[str],
            nice_to_have: List[str],
            domains: List[str],
            idf_must: Dict[str, float],
            idf_nice: Dict[str, float],
            top_k: int,
    ) -> Tuple[List[Any], str, List[Dict[str, Any]]]:
        if not gated_ids:
            return [], "", []
        params: List[Any] = list(gated_ids)
        must_params = list(must_have)
        nice_params = list(nice_to_have)
        domain_params = list(domains)
        sql = f"""
    WITH candidates AS (
      SELECT c.candidate_id,
             c.last_updated,
             SUM(CASE WHEN t.tag_type = 'tech' AND t.tag_key IN ({','.join(['?'] * len(must_params))}) THEN 1 ELSE 0 END) AS must_hit_count,
             SUM(CASE WHEN t.tag_type = 'tech' AND t.tag_key IN ({','.join(['?'] * len(nice_params))}) THEN 1 ELSE 0 END) AS nice_hit_count,
             MAX(CASE WHEN t.tag_type = 'domain' AND t.tag_key IN ({','.join(['?'] * len(domain_params))}) THEN 1 ELSE 0 END) AS domain_present
      FROM candidate_tag t
      JOIN candidate c ON c.candidate_id = t.candidate_id
      WHERE c.candidate_id IN ({','.join(['?'] * len(params))})
      GROUP BY c.candidate_id
    )
    SELECT * FROM candidates
    ORDER BY must_hit_count DESC, nice_hit_count DESC
    LIMIT ?
    """
        params = params + must_params + nice_params + domain_params + [top_k]
        plan = self.explain_query_plan(sql, params)
        rows = self.conn.execute(sql, params).fetchall()
        for row in rows:
            row["must_idf_sum"] = sum(idf_must.get(tag, 0.0) for tag in must_have)
            row["nice_idf_sum"] = sum(idf_nice.get(tag, 0.0) for tag in nice_to_have)
        return rows, sql.strip(), plan

    def explain_query_plan(self, sql: str, params: Iterable[Any]) -> List[Dict[str, Any]]:
        cur = self.conn.execute(f"EXPLAIN QUERY PLAN {sql}", tuple(params))
        columns = [col[0] for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_or_create_faiss_id(self, candidate_id: str) -> int:
        row = self.conn.execute(
            "SELECT faiss_id FROM faiss_id_map WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row:
            return int(row["faiss_id"])
        base = int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16], 16) & ((1 << 63) - 1)
        start = base
        h = base
        while True:
            try:
                self.conn.execute(
                    "INSERT INTO faiss_id_map(faiss_id, candidate_id) VALUES (?, ?)",
                    (h, candidate_id),
                )
                return int(h)
            except sqlite3.IntegrityError:
                existing = self.conn.execute(
                    "SELECT candidate_id FROM faiss_id_map WHERE faiss_id = ?",
                    (h,),
                ).fetchone()
                if existing and existing[0] == candidate_id:
                    return int(h)
                h = (h + 1) & ((1 << 63) - 1)
                if h == start:
                    raise RuntimeError("Unable to allocate FAISS ID")

    def get_full_candidate_context(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT summary_text, experience_text, tags_text, last_updated, location, seniority
            FROM candidate_doc
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()
        if not row:
            return None
        return {k: row[k] for k in row.keys()}
