FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY corpus ./corpus
COPY eval ./eval

# 索引不在镜像构建期生成。
# 默认嵌入后端是 ollama，构建时容器里没有 ollama 服务，连不上会失败。
# 改成容器启动时建（见 CMD）。
ENV SM_EMBED_BACKEND=${SM_EMBED_BACKEND:-hashing}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

# 启动时先建经营数据和索引，再起服务。
# 索引带嵌入器签名，检测到模型变了会自动重建，不用手动清库。
CMD ["sh", "-c", "python scripts/seed_ops_db.py && python scripts/build_index.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
