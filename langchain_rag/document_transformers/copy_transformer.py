# ruff: noqa: I001
import copy
from typing import Any, AsyncIterator, Iterator, Union

from langchain_rag.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
    to_async_iterator,
)
from langchain.schema import Document


class CopyDocumentTransformer(RunnableGeneratorDocumentTransformer):
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        yield from (copy.deepcopy(doc) for doc in documents)

    async def alazy_transform_documents(  # type: ignore
        self,
        documents: Union[AsyncIterator[Document], Iterator[Document]],
        **kwargs: Any
    ) -> AsyncIterator[Document]:
        if isinstance(documents, AsyncIterator):
            async_documents = documents
        else:
            async_documents = to_async_iterator(documents)
        async for doc in async_documents:
            yield copy.deepcopy(doc)
