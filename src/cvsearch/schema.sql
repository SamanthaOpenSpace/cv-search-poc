-- Core entities -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS candidate (
                                         candidate_id TEXT PRIMARY KEY,
                                         name         TEXT,
                                         location     TEXT,
                                         seniority    TEXT,
                                         last_updated TEXT
);

CREATE TABLE IF NOT EXISTS cv_file (
                                       id           INTEGER PRIMARY KEY AUTOINCREMENT,
                                       candidate_id TEXT NOT NULL,
                                       file_id      TEXT,
                                       mime         TEXT,
                                       sha256       TEXT,
                                       FOREIGN KEY(candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE
);

-- Experience with clarified CSV caches of tags (normalized tags live elsewhere)
CREATE TABLE IF NOT EXISTS experience (
                                          id               INTEGER PRIMARY KEY AUTOINCREMENT,
                                          candidate_id     TEXT NOT NULL,
                                          title            TEXT,
                                          company          TEXT,
                                          start            TEXT,
                                          end              TEXT,
                                          domain_tags_csv  TEXT,  -- was domain_tags
                                          tech_tags_csv    TEXT,  -- was tech
                                          highlights       TEXT,
                                          FOREIGN KEY(candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE
);

-- Optional legacy skills table (unchanged)
CREATE TABLE IF NOT EXISTS skill (
                                     id           INTEGER PRIMARY KEY AUTOINCREMENT,
                                     candidate_id TEXT NOT NULL,
                                     skill_raw    TEXT,
                                     skill_norm   TEXT,
                                     weight       REAL DEFAULT 1.0,
                                     FOREIGN KEY(candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE
);

-- Normalized tags -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS candidate_tag (
                                             candidate_id TEXT NOT NULL,
                                             tag_type     TEXT NOT NULL CHECK (tag_type IN ('role','tech','domain','seniority')),
                                             tag_key      TEXT NOT NULL,
                                             weight       REAL NOT NULL DEFAULT 1.0,
                                             FOREIGN KEY(candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE,
                                             UNIQUE(candidate_id, tag_type, tag_key)
);

CREATE TABLE IF NOT EXISTS experience_tag (
                                              experience_id INTEGER NOT NULL,
                                              tag_type      TEXT NOT NULL CHECK (tag_type IN ('tech','domain')),
                                              tag_key       TEXT NOT NULL,
                                              FOREIGN KEY(experience_id) REFERENCES experience(id) ON DELETE CASCADE,
                                              UNIQUE(experience_id, tag_type, tag_key)
);

-- Candidate-level document (search unit) -----------------------------------
CREATE TABLE IF NOT EXISTS candidate_doc (
                                             candidate_id    TEXT PRIMARY KEY, -- TEXT PK as requested; still has hidden rowid
                                             summary_text    TEXT,
                                             experience_text TEXT,
                                             tags_text       TEXT,
                                             last_updated    TEXT,   -- optional unindexed metadata for boosts
                                             location        TEXT,
                                             seniority       TEXT,
                                             FOREIGN KEY(candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE
);

-- REMOVED: embedding_doc table
-- This table is no longer needed as embeddings are stored in the FAISS file.

-- App-level configuration (e.g., vector_store_id)
CREATE TABLE IF NOT EXISTS app_config (
                                          key   TEXT PRIMARY KEY,
                                          value TEXT NOT NULL
);

-- REMOVED: vs_file_map table
-- This table is no longer needed as it was specific to the OpenAI Vector Store.


-- NEW: FAISS ID to Candidate ID Mapping Table
-- This replaces the cv_search_docs.json file.
CREATE TABLE IF NOT EXISTS faiss_id_map (
                                            faiss_id     INTEGER PRIMARY KEY, -- The integer index ID from the FAISS file
                                            candidate_id TEXT NOT NULL,
                                            FOREIGN KEY(candidate_id) REFERENCES candidate(candidate_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_faiss_map_candidate ON faiss_id_map(candidate_id);


-- Helpful indexes -----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_skill_candidate               ON skill(candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_tag_candidate       ON candidate_tag(candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_tag_key             ON candidate_tag(tag_key);
CREATE INDEX IF NOT EXISTS idx_experience_tag_experience     ON experience_tag(experience_id);
CREATE INDEX IF NOT EXISTS idx_experience_tag_key            ON experience_tag(tag_key);

-- REMOVED: idx_embed_doc_candidate_model (table is gone)

-- --- ADDED INDICES ---
CREATE INDEX IF NOT EXISTS idx_ctag_type_key_cid ON candidate_tag(tag_type, tag_key, candidate_id);
CREATE INDEX IF NOT EXISTS idx_ctag_cid_type ON candidate_tag(candidate_id, tag_type);
-- --- END ADDED INDICES ---