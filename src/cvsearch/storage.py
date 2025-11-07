from __future__ import annotations
import sqlite3
import json
import math
from typing import Dict, Any, Optional, List, Tuple, Iterable

from src.cvsearch.settings import Settings


class CVDatabase:
    """
    Centralized class for all SQLite database interactions.
    Manages the connection and exposes methods for all queries.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_path = str(settings.db_path)
        self.schema_file = str(settings.schema_file)
        self.conn = self._get_db()

    def _get_db(self) -> sqlite3.Connection:
        """Creates and returns a new DB connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def commit(self):
        """Commits the current transaction."""
        self.conn.commit()

    def initialize_schema(self):
        """(Moved from config.py)"""
        with open(self.schema_file, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.commit()

    def check_tables(self) -> List[str]:
        """Helper for main.py check-db command."""
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
        return [r[0] for r in rows]

    def check_fts(self) -> str:
        """Helper for main.py check-db command."""
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts_test USING fts5(x);")
            self.conn.execute("DROP TABLE IF EXISTS __fts_test;")
            return "available"
        except sqlite3.OperationalError as e:
            return f"not available: {e}"

    def set_app_config(self, key: str, value: str) -> None:
        """(Moved from api_client.py)"""
        self.conn.execute(
            """
            INSERT INTO app_config(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def get_app_config(self, key: str) -> Optional[str]:
        """(Moved from api_client.py)"""
        row = self.conn.execute("SELECT value FROM app_config WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def remove_candidate_derived(self, candidate_id: str) -> None:
        """(Moved from ingest.py)"""
        self.conn.execute("DELETE FROM faiss_id_map WHERE candidate_id = ?", (candidate_id,))
        self.conn.execute("DELETE FROM candidate_doc WHERE candidate_id = ?", (candidate_id,))
        exp_ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM experience WHERE candidate_id = ?", (candidate_id,)
        ).fetchall()]
        if exp_ids:
            self.conn.executemany("DELETE FROM experience_tag WHERE experience_id = ?", [(i,) for i in exp_ids])
        self.conn.execute("DELETE FROM experience WHERE candidate_id = ?", (candidate_id,))
        self.conn.execute("DELETE FROM candidate_tag WHERE candidate_id = ?", (candidate_id,))

    def upsert_candidate(self, cv: Dict[str, Any]) -> None:
        """(Moved from ingest.py)"""
        self.conn.execute(
            """
            INSERT INTO candidate(candidate_id, name, location, seniority, last_updated)
            VALUES(?,?,?,?,?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                                                    name=excluded.name,
                                                    location=excluded.location,
                                                    seniority=excluded.seniority,
                                                    last_updated=excluded.last_updated
            """,
            (
                cv["candidate_id"],
                cv.get("name", "[redacted]"),
                cv.get("location", ""),
                cv.get("seniority", ""),
                cv.get("last_updated", ""),
            ),
        )

    def insert_experiences_and_tags(self, candidate_id: str, experiences: List[Dict[str, Any]], domain_tags_list: List[str], tech_tags_list: List[str]) -> List[int]:
        """(Refactored from ingest.py)"""
        exp_ids: List[int] = []

        exp_tech_tags_to_insert = []
        exp_domain_tags_to_insert = []

        for i, exp in enumerate(experiences or []):
            domain_tags = domain_tags_list[i]
            tech_tags = tech_tags_list[i]

            cur = self.conn.execute(
                """
                INSERT INTO experience(candidate_id, title, company, start, end, domain_tags_csv, tech_tags_csv, highlights)
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

            for t in tech_tags:
                exp_tech_tags_to_insert.append((exp_id, "tech", t))
            for d in domain_tags:
                exp_domain_tags_to_insert.append((exp_id, "domain", d))

        self.conn.executemany(
            "INSERT INTO experience_tag(experience_id, tag_type, tag_key) VALUES (?,?,?) ON CONFLICT(experience_id, tag_type, tag_key) DO NOTHING",
            exp_tech_tags_to_insert
        )
        self.conn.executemany(
            "INSERT INTO experience_tag(experience_id, tag_type, tag_key) VALUES (?,?,?) ON CONFLICT(experience_id, tag_type, tag_key) DO NOTHING",
            exp_domain_tags_to_insert
        )

        return exp_ids

    def upsert_candidate_tags(self, candidate_id: str,
                              role_tags: List[str],
                              tech_tags_top: List[str],
                              seniority: str,
                              domain_rollup: List[str]) -> None:
        """(Moved from ingest.py)"""
        tags_to_insert = []

        for r in role_tags:
            if r:
                tags_to_insert.append((candidate_id, "role", r, 2.0))
        for t in tech_tags_top:
            if t:
                tags_to_insert.append((candidate_id, "tech", t, 1.5))
        if seniority:
            tags_to_insert.append((candidate_id, "seniority", seniority, 1.0))
        for d in domain_rollup:
            if d:
                tags_to_insert.append((candidate_id, "domain", d, 1.0))

        self.conn.executemany(
            """
            INSERT INTO candidate_tag(candidate_id, tag_type, tag_key, weight)
            VALUES (?,?,?,?)
            ON CONFLICT(candidate_id, tag_type, tag_key)
                DO UPDATE SET weight = excluded.weight
            """,
            tags_to_insert
        )

    def upsert_candidate_doc(self, candidate_id: str,
                             summary_text: str,
                             experience_text: str,
                             tags_text: str,
                             last_updated: str,
                             location: str,
                             seniority: str) -> None:
        """(Moved from ingest.py)"""
        self.conn.execute(
            """
            INSERT INTO candidate_doc(candidate_id, summary_text, experience_text, tags_text, last_updated, location, seniority)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                                                    summary_text=excluded.summary_text,
                                                    experience_text=excluded.experience_text,
                                                    tags_text=excluded.tags_text,
                                                    last_updated=excluded.last_updated,
                                                    location=excluded.location,
                                                    seniority=excluded.seniority
            """,
            (candidate_id, summary_text, experience_text, tags_text, last_updated, location, seniority),
        )

    def get_all_candidate_docs_for_embedding(self) -> List[sqlite3.Row]:
        """
        Fetches all candidate_id and text fields from candidate_doc.
        Used to build the global FAISS index.
        """
        return self.conn.execute(
            """
            SELECT candidate_id, summary_text, experience_text, tags_text
            FROM candidate_doc
            """
        ).fetchall()

    def explain_query_plan(self, sql: str, params: Tuple[Any, ...] | List[Any]) -> List[Dict[str, Any]]:
        """(Moved from search.py)"""
        try:
            return [dict(r) for r in self.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()]
        except Exception:
            return []

    def compute_idf(self, tags: List[str], tag_type: str) -> Dict[str, float]:
        """(Moved from search.py)"""
        if not tags:
            return {}
        N_row = self.conn.execute("SELECT COUNT(*) AS n FROM candidate").fetchone()
        N = N_row["n"] if N_row else 0

        placeholders = ",".join(["?"] * len(tags))
        rows = self.conn.execute(
            f"""
            SELECT tag_key AS tag, COUNT(DISTINCT candidate_id) AS df
            FROM candidate_tag
            WHERE tag_type = ?
              AND tag_key IN ({placeholders})
            GROUP BY tag_key
            """,
            [tag_type, *tags],
        ).fetchall()
        df_map = {r["tag"]: r["df"] for r in rows}
        idf = {}
        for t in tags:
            df = df_map.get(t, 0)
            idf[t] = math.log((N + 1.0) / (df + 1.0)) + 1.0
        return idf

    def get_gated_candidate_ids(self, seat: Dict[str, Any]) -> Tuple[List[str], str, List[Dict[str, Any]]]:
        """(Moved from search.py, uses helper from retrieval.py)"""

        def _get_allowed_seniorities(seat_seniority: str) -> Tuple[str, ...]:
            ladder = ("junior", "mid", "senior", "staff", "principal")
            if seat_seniority not in ladder:
                return ("senior",)
            idx = ladder.index(seat_seniority)
            return ladder[idx:]

        role = seat["role"]
        allowed_sens = tuple(_get_allowed_seniorities(seat["seniority"]))

        sql = f"""
    WITH gated AS (
      SELECT c.candidate_id
      FROM candidate c
      WHERE EXISTS (
        SELECT 1
        FROM candidate_tag t
        WHERE t.candidate_id = c.candidate_id
          AND t.tag_type = 'role'
          AND t.tag_key  = ?
      )
      AND EXISTS (
        SELECT 1
        FROM candidate_tag t
        WHERE t.candidate_id = c.candidate_id
          AND t.tag_type = 'seniority'
          AND t.tag_key  IN ({",".join(["?"]*len(allowed_sens))})
      )
    )
    SELECT candidate_id FROM gated
    """
        params = [role] + list(allowed_sens)
        plan = self.explain_query_plan(sql, params)
        rows = self.conn.execute(sql, params).fetchall()
        return [r["candidate_id"] for r in rows], sql.strip(), plan

    def rank_weighted_set(self,
                          gated_ids: List[str],
                          must_have: List[str],
                          nice_to_have: List[str],
                          domains: List[str],
                          idf_must: Dict[str, float],
                          idf_nice: Dict[str, float],
                          alpha: float = 2.0,
                          beta: float = 1.0,
                          gamma: float = 0.3,
                          delta: float = 0.5,
                          top_k: int = 10) -> Tuple[List[sqlite3.Row], str, List[Dict[str, Any]]]:
        """(Moved from search.py)"""
        if not gated_ids:
            return [], "", []

        self.conn.execute("DROP TABLE IF EXISTS tmp_gated")
        self.conn.execute("CREATE TEMP TABLE tmp_gated(candidate_id TEXT PRIMARY KEY)")
        self.conn.executemany("INSERT INTO tmp_gated(candidate_id) VALUES (?)", [(cid,) for cid in gated_ids])

        if must_have:
            must_vals_sql = ",".join(["(?, ?)"] * len(must_have))
            cte_must = f"must_tags(tag, idf) AS (VALUES {must_vals_sql})"
            must_params: List[Any] = [p for t in must_have for p in (t, idf_must.get(t, 0.0))]
        else:
            cte_must = "must_tags(tag, idf) AS (SELECT NULL, NULL WHERE 0)"
            must_params = []
        if nice_to_have:
            nice_vals_sql = ",".join(["(?, ?)"] * len(nice_to_have))
            cte_nice = f"nice_tags(tag, idf) AS (VALUES {nice_vals_sql})"
            nice_params: List[Any] = [p for t in nice_to_have for p in (t, idf_nice.get(t, 0.0))]
        else:
            cte_nice = "nice_tags(tag, idf) AS (SELECT NULL, NULL WHERE 0)"
            nice_params = []
        if domains:
            dom_vals_sql = ",".join(["(?)"] * len(domains))
            cte_domains = f"domains(tag) AS (VALUES {dom_vals_sql})"
            dom_params: List[Any] = list(domains)
        else:
            cte_domains = "domains(tag) AS (SELECT NULL WHERE 0)"
            dom_params = []

        cte_scalars = "scalars AS (SELECT ? AS alpha, ? AS beta, ? AS gamma, ? AS delta, ? AS M)"
        domain_present_sql = "0"
        if domains:
            domain_present_sql = """CASE WHEN EXISTS (
                SELECT 1 FROM candidate_tag t
                JOIN domains d ON d.tag = t.tag_key
                WHERE t.candidate_id = c.candidate_id AND t.tag_type='domain'
            ) THEN 1 ELSE 0 END"""

        nice_hit_sql = "0"
        nice_idf_sum_sql = "0.0"
        if nice_to_have:
            nice_hit_sql = """(SELECT COUNT(*) FROM candidate_tag t
                               JOIN nice_tags n ON n.tag = t.tag_key
                               WHERE t.candidate_id = c.candidate_id AND t.tag_type='tech')"""
            nice_idf_sum_sql = """(SELECT SUM(n.idf) FROM nice_tags n
                                   WHERE EXISTS (SELECT 1 FROM candidate_tag t
                                                 WHERE t.candidate_id = c.candidate_id
                                                   AND t.tag_type='tech'
                                                   AND t.tag_key = n.tag))"""

        must_hit_sql = "0"
        must_idf_sum_sql = "0.0"
        if must_have:
            must_hit_sql = """(SELECT COUNT(*) FROM candidate_tag t
                               JOIN must_tags m ON m.tag = t.tag_key
                               WHERE t.candidate_id = c.candidate_id AND t.tag_type='tech')"""
            must_idf_sum_sql = """(SELECT SUM(m.idf) FROM must_tags m
                                   WHERE EXISTS (SELECT 1 FROM candidate_tag t
                                                 WHERE t.candidate_id = c.candidate_id
                                                   AND t.tag_type='tech'
                                                   AND t.tag_key = m.tag))"""

        sql = f"""
    WITH
      {cte_must},
      {cte_nice},
      {cte_domains},
      {cte_scalars},
      per_candidate AS (
        SELECT
          c.candidate_id,
          d.last_updated,
          {must_hit_sql}            AS must_hit_count,
          {must_idf_sum_sql}        AS must_idf_sum,
          {nice_hit_sql}            AS nice_hit_count,
          {nice_idf_sum_sql}        AS nice_idf_sum,
          {domain_present_sql}      AS domain_present
        FROM tmp_gated c
        JOIN candidate_doc d ON d.candidate_id = c.candidate_id
      )
    SELECT
      candidate_id,
      (alpha * (CASE WHEN M > 0 THEN (must_hit_count / M) ELSE 0 END)
       + beta  * COALESCE(must_idf_sum, 0.0)
       + gamma * COALESCE(nice_idf_sum, 0.0)
       + delta * domain_present
      ) AS score,
      must_hit_count, must_idf_sum, nice_hit_count, nice_idf_sum, domain_present,
      last_updated
    FROM per_candidate, scalars
    ORDER BY score DESC, last_updated DESC, candidate_id ASC
    LIMIT ?
    """
        params: List[Any] = []
        params += must_params
        params += nice_params
        params += dom_params
        params += [alpha, beta, gamma, delta, float(len(must_have))]
        params += [top_k]

        plan = self.explain_query_plan(sql, params)
        rows = self.conn.execute(sql, params).fetchall()
        self.conn.execute("DROP TABLE IF EXISTS tmp_gated")
        return rows, sql.strip(), plan

    def fetch_tag_hits(self,
                       candidate_ids: List[str],
                       tags: List[str]) -> Dict[str, Dict[str, bool]]:
        """(Moved from search.py)"""
        if not candidate_ids or not tags:
            return {}

        placeholders_c = ",".join(["?"] * len(candidate_ids))
        placeholders_t = ",".join(["?"] * len(tags))
        sql = f"""
    SELECT candidate_id, tag_key
    FROM candidate_tag
    WHERE tag_type='tech'
      AND candidate_id IN ({placeholders_c})
      AND tag_key IN ({placeholders_t})
    """
        rows = self.conn.execute(sql, list(candidate_ids) + list(tags)).fetchall()
        res: Dict[str, Dict[str, bool]] = {cid: {t: False for t in tags} for cid in candidate_ids}
        for r in rows:
            res[r["candidate_id"]][r["tag_key"]] = True
        return res

    def get_full_candidate_context(self, candidate_id: str) -> Dict[str, Any] | None:
        """
        Fetches the pre-compiled text context for a single candidate.
        """
        row = self.conn.execute(
            "SELECT summary_text, experience_text, tags_text FROM candidate_doc WHERE candidate_id = ?",
            (candidate_id,)
        ).fetchone()

        if row:
            return dict(row)
        return None

    def clear_faiss_id_map(self) -> None:
        """
        Deletes all rows from the faiss_id_map table.
        """
        self.conn.execute("DELETE FROM faiss_id_map")

    def insert_faiss_id_map_batch(self, mappings: List[Tuple[int, str]]) -> None:
        """
        Inserts a batch of (faiss_id, candidate_id) mappings.
        """
        if not mappings:
            return

        self.conn.executemany(
            "INSERT INTO faiss_id_map (faiss_id, candidate_id) VALUES (?, ?)",
            mappings
        )

    def get_candidate_ids_from_faiss_ids(self, faiss_ids: List[int]) -> Dict[int, str]:
        """
        Efficiently retrieves a mapping of {faiss_id: candidate_id} for a given
        list of faiss_ids.
        """
        if not faiss_ids:
            return {}

        placeholders = ",".join(["?"] * len(faiss_ids))
        sql = f"""
            SELECT faiss_id, candidate_id
            FROM faiss_id_map
            WHERE faiss_id IN ({placeholders})
        """
        rows = self.conn.execute(sql, faiss_ids).fetchall()
        return {row["faiss_id"]: row["candidate_id"] for row in rows}

    # --- START NEW METHOD ---

    def get_or_create_faiss_id(self, candidate_id: str) -> int:
        """
        Atomically retrieves the 64-bit int FAISS ID for a candidate.
        If the candidate is new, it generates a new ID by incrementing
        the maximum existing ID.

        This method MUST be called from within an existing transaction
        to ensure atomicity of the SELECT MAX / INSERT.
        """
        # 1. Check if ID already exists
        cur = self.conn.execute(
            "SELECT faiss_id FROM faiss_id_map WHERE candidate_id = ?",
            (candidate_id,)
        )
        row = cur.fetchone()

        if row:
            return int(row["faiss_id"])

        # 2. If not, create a new one
        # This is safe from race conditions ONLY because the
        # calling function (pipeline.upsert_cvs) holds a transaction.
        cur = self.conn.execute("SELECT MAX(faiss_id) FROM faiss_id_map")
        row = cur.fetchone()

        new_id = (int(row[0]) if row and row[0] is not None else 0) + 1

        self.conn.execute(
            "INSERT INTO faiss_id_map (faiss_id, candidate_id) VALUES (?, ?)",
            (new_id, candidate_id)
        )

        return new_id
    # --- END NEW METHOD ---