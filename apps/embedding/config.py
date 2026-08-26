"""Embedding-app settings.

Re-exports the shared pydantic-settings config (apps/common/config.py).
The app never reads os.environ directly — everything comes from Settings,
which layers process env over the repo-root `.env` file.
"""

from __future__ import annotations

from apps.common.config import DB_PATH, settings

__all__ = [
    "DB_PATH",
    "settings",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "BATCH_SIZE",
    "RERANK_ENABLED",
    "TOP_K",
]

EMBEDDING_MODEL = settings.openai_embedding_model
EMBEDDING_DIM = settings.embedding_dim
BATCH_SIZE = settings.embedding_batch_size
RERANK_ENABLED = settings.rerank_enabled
TOP_K = settings.top_k
