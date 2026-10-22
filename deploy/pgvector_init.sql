-- StoreMind pgvector 迁移 schema
--
-- 当前代码跑在 SQLite 上（chunk 存 float32 blob，检索在 Python 侧算余弦）。
-- 这份 schema 是迁移目标：向量进库、检索下推给 PG、能跟业务表 JOIN。
--
-- 迁移触发条件：chunk 数过 10 万，或需要「向量检索 + 业务条件过滤」同时做的时候。
-- 在那之前上 PG 只是多一个要运维的组件。
--
-- 用法：
--   docker compose -f deploy/docker-compose.yml --profile pg up -d
--   psql -h 127.0.0.1 -U storemind -d storemind -f deploy/pgvector_init.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- bge-m3 输出 1024 维。换模型必须改这个维度并重建索引。
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    doc_title    TEXT NOT NULL,
    section_path TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    embed_text   TEXT NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    embedding    vector(1024) NOT NULL,
    tsv          tsvector,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- HNSW 比 IVFFlat 更适合「边写边查」的场景（IVFFlat 需要先有数据才能训练列表）
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

-- 中文全文检索：PG 自带分词对中文不友好，生产建议装 zhparser 或 pg_jieba。
-- 这里先用 simple 配置兜底，配合应用侧的字符 bigram 分词写入 tsv。
-- 若要开箱可用，把下面这行替换成：
--   CREATE TEXT SEARCH CONFIGURATION zh (COPY = simple);
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('simple', NEW.embed_text);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks;
CREATE TRIGGER trg_chunks_tsv
    BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();

-- 检索示例：向量召回 top 20
--   SELECT id, section_path, 1 - (embedding <=> $1) AS score
--   FROM chunks ORDER BY embedding <=> $1 LIMIT 20;
--
-- 混合检索（向量 + 关键词）在应用层用 RRF 融合两路结果，
-- 不在这里写 SQL，因为 RRF 的 k 值和权重需要跟着评估集调，
-- 放进数据库就改不动了。
