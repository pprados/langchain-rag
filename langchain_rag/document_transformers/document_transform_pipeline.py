from itertools import chain
from typing import Any, AsyncIterator, Iterator, Sequence, cast

from langchain_core.documents import BaseDocumentTransformer, Document

from .document_transformers import BATCH_SIZE, async_batched
from .lazy_document_transformer import (
    LazyDocumentTransformer,
)


class DocumentTransformerPipeline(LazyDocumentTransformer):

    """List of document transformers that are chained together and run in sequence."""

    def __init__(
        self,
        transformers: Sequence[BaseDocumentTransformer],
        batch_size: int = BATCH_SIZE,
    ):
        self.transformers = transformers
        self.batch_size = batch_size

    def _lazy_transform_documents_with_transformer(
        self,
        documents: Iterator[Document],
        transformer: BaseDocumentTransformer,
        **kwargs: Any,
    ) -> Iterator[Document]:
        lazy_results = []
        for _document in documents:
            if hasattr(transformer, "lazy_transform_documents"):
                lazy_results.append(
                    transformer.lazy_transform_documents([_document], **kwargs)
                )
            else:
                lazy_results.append(
                    iter(transformer.transform_documents([_document], **kwargs))
                )
        return chain(*lazy_results)

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        if not self.transformers:
            return iter(documents)
        for _transformer in self.transformers:
            documents = self._lazy_transform_documents_with_transformer(
                documents, transformer=_transformer
            )
        return documents

    async def _alazy_transform_documents_with_transformer(
        self,
        documents: AsyncIterator[Document],
        transformer: BaseDocumentTransformer,
        **kwargs: Any,
    ) -> AsyncIterator[Document]:
        if hasattr(transformer, "alazy_transform_documents"):
            async for doc in transformer.alazy_transform_documents(documents, **kwargs):
                yield doc

        else:
            async for batch in async_batched(documents, self.batch_size):
                sync_batch = cast(Sequence[Document], [doc async for doc in batch])
                for doc in transformer.transform_documents(sync_batch, **kwargs):
                    yield doc

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        for _transformer in self.transformers:
            async for doc in self._alazy_transform_documents_with_transformer(
                documents, transformer=_transformer
            ):
                yield doc
