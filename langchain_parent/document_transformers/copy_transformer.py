from typing import Sequence, Any, Iterator

from langchain.schema import Document
from langchain_parent.document_transformers.runnable_document_transformer import \
    RunnableGeneratorDocumentTransformer


class CopyDocumentTransformer(RunnableGeneratorDocumentTransformer):
    def lazy_transform_documents(
            self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        yield from (doc for doc in documents)

