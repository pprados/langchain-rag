import copy
from typing import Any, AsyncIterator, Callable, Iterator, Sequence, Union

import langchain

from langchain_rag.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer, RunnableDocumentTransformer,
)


class _LazyTransformer(RunnableGeneratorDocumentTransformer):
    """Implementation of a runnable transformer, with lazy transformation"""

    fn: Callable[[Any], str]

    def lazy_transform_documents(
        self, documents: Iterator[langchain.schema.Document], **kwargs: Any
    ) -> Iterator[langchain.schema.Document]:
        return (
            langchain.schema.Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )
            for doc in documents
        )

    async def _alazy_transform_documents(  # type:ignore
        self,
        documents: Union[
            AsyncIterator[langchain.schema.Document],
            Iterator[langchain.schema.Document],
        ],
        **kwargs: Any
    ) -> AsyncIterator[langchain.schema.Document]:
        async for doc in documents:
            yield langchain.schema.Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )


class _Transformer(RunnableDocumentTransformer):
    """Implementation of a runnable transformer, without lazy transformation"""

    fn: Callable[[Any], str]

    def transform_documents(
        self, documents: Sequence[langchain.schema.Document], **kwargs: Any
    ) -> Sequence[langchain.schema.Document]:
        return [
            langchain.schema.Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata),
            )
            for doc in documents
        ]

    async def atransform_documents(
        self, documents: Sequence[langchain.schema.Document], **kwargs: Any
    ) -> Sequence[langchain.schema.Document]:
        return self.transform_documents(documents=documents, **kwargs)


class UpperLazyTransformer(_LazyTransformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.upper, **kwargs)


class UpperTransformer(_Transformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.upper, **kwargs)


class LowerLazyTransformer(_LazyTransformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.lower, **kwargs)


class LowerTransformer(_Transformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.lower, **kwargs)
