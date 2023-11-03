import copy
from typing import Any, Iterator, Union, AsyncIterator, cast

from langchain.schema import Document

from langchain_rag.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer, to_async_iterator,
)


class CopyDocumentTransformer(RunnableGeneratorDocumentTransformer):
    def lazy_transform_documents(
            self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        yield from (copy.deepcopy(doc) for doc in documents)

    async def alazy_transform_documents(
            self, documents: Union[AsyncIterator[Document], Iterator[Document]],
            **kwargs: Any
    ) -> AsyncIterator[Document]:
        if isinstance(documents, AsyncIterator):
            async_documents = documents
        else:
            async_documents = to_async_iterator(documents)
        return async_documents
