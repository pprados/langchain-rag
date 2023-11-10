# ruff: noqa: I001
from typing import List

import pytest
from langchain.schema.embeddings import Embeddings

from langchain_rag.vectorstores.wrapper_vectorstore import WrapperVectorStore
import requests


class _FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def _is_api_accessible(url: str) -> bool:
    try:
        response = requests.get(url)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.requires("faiss")
def test_from_text() -> None:
    from langchain.vectorstores import FAISS

    wrapper_vs = WrapperVectorStore.from_texts(
        vectorstore_cls=FAISS,
        texts=["hello", "world"],
        embedding=_FakeEmbeddings(),
        metadatas=[{"id": 1}, {"id": 2}],
    )
    assert isinstance(wrapper_vs.embeddings, _FakeEmbeddings)
    assert isinstance(wrapper_vs.vectorstore, FAISS)


@pytest.mark.requires("chromadb")
@pytest.mark.skipif(
    not _is_api_accessible("http://localhost:8000/api/v1/heartbeat"),
    reason="API not accessible",
)
def test_self_query_with_wrapper_vectorstore() -> None:
    from langchain.vectorstores.chroma import Chroma
    from langchain.retrievers.self_query.base import (
        _get_builtin_translator,
        ChromaTranslator,
    )

    wrapper_vs = WrapperVectorStore(vectorstore=Chroma())
    result = _get_builtin_translator(wrapper_vs)
    assert isinstance(result, ChromaTranslator)
