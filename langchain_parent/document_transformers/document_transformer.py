from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Iterator, Sequence, Any

from langchain.schema import BaseDocumentTransformer, Document


class _AsyncIterator_from_Iterator(AsyncIterator):

    def __init__(self, iterator):
        self.iterator = iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        return next(self.iterator)


class GeneratorBaseDocumentTransformer(BaseDocumentTransformer):
    # this is a transition class to add a lazy_transform_documents variations
    # Later, it may be integrated in ̀BaseDocumentTransformer`
    def transform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return list(self.lazy_transform_documents(iter(documents), **kwargs))

    async def atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        # async for doc in self.alazy_transform_documents(iter(documents), **kwargs):
        #     print(doc)
        return [doc async for doc in
                self.alazy_transform_documents(iter(documents), **kwargs)]

    @abstractmethod
    def lazy_transform_documents(
            self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an interator of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        # Default implementation. Not realy lazy.
        return iter(self.transform_documents(list(documents)))

    async def alazy_transform_documents(
            self, documents: Iterator[Document],
            **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An interator of transformed Documents.
        """
        # Default implementation. Not realy lazy.
        result = await self.atransform_documents(list(documents))
        return _AsyncIterator_from_Iterator(iter(result))
