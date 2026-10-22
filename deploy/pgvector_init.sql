-- StoreMind PostgreSQL + pgvector 参考 schema
--
-- 运行时 schema 由 app/rag/pg_store.py 创建，维度取自 SM_EMBED_DIM
-- （hashing 512 / bge-m3 1024）。这个文件用于展示生产目标，不是启动依赖：
-- docker-compose 不再把它挂到 /docker-entrypoint-initdb.d。
--
-- 为什么 BM25 不在这里：项目保留了应用层的字符 bigram BM25（chunk_terms 表 +
-- Python 打分），换成 PG 的 tsvector 会改变词法召回，迁移前后指标就不可比了。
-- 中文全文检索的生产方案是 zhparser / pg_jieba，不是 simple 配置。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);

-- bge-m3 输出 1024 维；hashing 是 512 维。
-- HNSW 要求固定维度，所以换嵌入后端必须重建这张表（PgStore 会自动检测）。
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    doc_title    TEXT NOT NULL,
    section_path TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    embed_text   TEXT NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    embedding    vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- HNSW 比 IVFFlat 更适合「边写边查」：IVFFlat 需要先有数据才能训练列表。
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 应用层 BM25 的词表，和 SQLite 版同构。
CREATE TABLE IF NOT EXISTS chunk_terms (
    chunk_id BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    term     TEXT NOT NULL,
    tf       INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, term)
);
CREATE INDEX IF NOT EXISTS idx_terms_term ON chunk_terms(term);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 检索示例：向量召回 top 20
--   SELECT id, section_path, 1 - (embedding <=> $1) AS score
--   FROM chunks WHERE dim = 1024 ORDER BY embedding <=> $1 LIMIT 20;
--
-- 混合检索（向量 + BM25）在应用层用 RRF 融合两路结果，
-- 不在这里写 SQL，因为 RRF 的 k 值和权重需要跟着评估集调。
