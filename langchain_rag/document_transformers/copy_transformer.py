import copy
from typing import Any, AsyncIterator, Iterator

from langchain.schema import Document

from langchain_rag.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
)


class CopyDocumentTransformer(RunnableGeneratorDocumentTransformer):
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        yield from (copy.deepcopy(doc) for doc in documents)

    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        async for doc in documents:
            yield copy.deepcopy(doc)
