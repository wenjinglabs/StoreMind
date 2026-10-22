FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 依赖单独一层：改代码不会触发重新装包
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY corpus ./corpus
COPY eval ./eval

# 默认零依赖：不需要 GPU、不需要宿主机 ollama，镜像自己就能建索引并作答。
# 要接本地模型，在 compose 里覆盖 SM_EMBED_BACKEND=ollama / SM_LLM_BACKEND=ollama，
# 并把 SM_OLLAMA_BASE 指向宿主机（见 docker-compose.yml）。
ENV SM_EMBED_BACKEND=hashing \
    SM_LLM_BACKEND=none

EXPOSE 8000 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

# 启动时先建经营数据和索引，再起服务。
# 索引带嵌入器签名，检测到模型变了会自动重建，不用手动清库。
CMD ["sh", "-c", "python scripts/seed_ops_db.py && python scripts/build_index.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
