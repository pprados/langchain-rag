from abc import abstractmethod
from typing import Iterator, Sequence, Any

from langchain.schema import BaseDocumentTransformer, Document


class GeneratorBaseDocumentTransformer(BaseDocumentTransformer):
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return list(self.lazy_transform_documents(iter(documents),**kwargs))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return list(await self.alazy_transform_documents(iter(documents),**kwargs))

    @abstractmethod
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
        # Default implementation
        return iter(self.transform_documents(list(documents)))

    async def alazy_transform_documents(
        self, documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
        return iter(await self.atransform_documents(list(documents)))

