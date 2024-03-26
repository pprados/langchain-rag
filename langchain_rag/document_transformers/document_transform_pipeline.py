from itertools import chain
from typing import Any, AsyncIterator, Iterator, Sequence, cast, List

from langchain_core.documents import BaseDocumentTransformer, Document

from .runnable_document_transformer import (
    _RunnableGeneratorDocumentTransformer,
    to_async_iterator,
)

def achain(*args) -> AsyncIterator:
    class ChainAsyncIterator(AsyncIterator):
        def __init__(self, *args):
            self.args = args
            self.index = 0

        async def __anext__(self):
            if self.index < len(self.args):
                result = await self.args[self.index]
                self.index += 1
                return result
            else:
                raise StopAsyncIteration
    return ChainAsyncIterator(*args)


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
        lazy_results:List[AsyncIterator]
        if hasattr(transformer, "lazy_transform_documents"):
            # Version classique iterable
            async for input_doc in documents:
                async for doc in transformer.alazy_transform_documents([input_doc], **kwargs):
                       yield doc

            # Version chain
            # FIXME: trouver comment éviter des étapes.
            # lazy_results = [
            #     cast(
            #         _RunnableGeneratorDocumentTransformer, transformer
            #     ).alazy_transform_documents(documents, **kwargs)
            # ]

        else:  # FIXME: a valider
            lazy_results = [
                to_async_iterator(iter(await transformer.atransform_documents([_document], **kwargs)))
                    async for _document in documents
            ]
        # lazy_results = []
        # async for _document in documents:
        #     if hasattr(transformer, "lazy_transform_documents"):
        #         lazy_results.append(
        #             await cast(
        #                 _RunnableGeneratorDocumentTransformer, transformer
        #             ).alazy_transform_documents([_document], **kwargs)
        #         )
        #     else:
        #         lazy_results.append(
        #             iter(await transformer.atransform_documents([_document], **kwargs))
        #         )
        # return achain(*lazy_results)

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        for _transformer in self.transformers:
            async for doc in self._alazy_transform_documents_with_transformer(
                documents, transformer=_transformer
            ):
                yield doc
            # yield await self._alazy_transform_documents_with_transformer(
            #     documents, transformer=_transformer
            # )
