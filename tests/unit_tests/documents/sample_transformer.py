"""
Some very simple transformer (lower, upper), lazy and compatible with LCEL.
"""

import copy
from typing import Any, AsyncIterator, Callable, Iterator

from langchain_core.documents import Document

from langchain_rag.document_transformers.lazy_document_transformer import (
    LazyDocumentTransformer,
)


class _LazyTransformer(LazyDocumentTransformer):
    """Implementation of a runnable transformer, with lazy transformation"""

    def __init__(self, fn: Callable[[Any], str]):
        self.fn = fn

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        return (
            Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )
            for doc in documents
        )

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        async for doc in documents:
            yield Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )


class LowerLazyTransformer(_LazyTransformer):
    def __init__(self) -> None:
        super().__init__(fn=str.lower)


class UpperLazyTransformer(_LazyTransformer):
    def __init__(self) -> None:
        super().__init__(fn=str.upper)
