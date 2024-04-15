from itertools import chain
from typing import Any, AsyncIterator, Iterator, Sequence, cast

from langchain_core.documents import BaseDocumentTransformer, Document

from .runnable_document_transformer import (
    _RunnableGeneratorDocumentTransformer,
)


class DocumentTransformerPipeline(_RunnableGeneratorDocumentTransformer):
    class Config:
        arbitrary_types_allowed = True

    transformers: Sequence[BaseDocumentTransformer]
    """List of document transformers that are chained together and run in sequence."""

    # def __init__(self, transformers: Sequence[BaseDocumentTransformer]):
    #     self.transformers = transformers

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
        async for _document in documents:
            if hasattr(transformer, "alazy_transform_documents"):
                async for doc in cast(
                    _RunnableGeneratorDocumentTransformer, transformer
                ).alazy_transform_documents([_document], **kwargs):
                    yield doc
            else:
                docs = await transformer.atransform_documents([_document], **kwargs)
                for doc in docs:
                    yield doc

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        for _transformer in self.transformers:
            async for doc in self._alazy_transform_documents_with_transformer(
                documents, transformer=_transformer
            ):
                yield doc
