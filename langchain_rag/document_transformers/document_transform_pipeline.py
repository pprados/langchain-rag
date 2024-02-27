from itertools import chain
from typing import Any, AsyncIterator, Iterator, Sequence, cast

from langchain_core.documents import BaseDocumentTransformer, Document

from .runnable_document_transformer import (
    _RunnableGeneratorDocumentTransformer,
    to_async_iterator,
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
        lazy_results = []
        async for _document in documents:
            if hasattr(transformer, "lazy_transform_documents"):
                lazy_results.append(
                    await cast(
                        _RunnableGeneratorDocumentTransformer, transformer
                    ).alazy_transform_documents([_document], **kwargs)
                )
            else:
                lazy_results.append(
                    iter(await transformer.atransform_documents([_document], **kwargs))
                )
        return to_async_iterator(chain(*lazy_results))

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        for _transformer in self.transformers:
            documents = await self._alazy_transform_documents_with_transformer(
                documents, transformer=_transformer
            )
        return documents
