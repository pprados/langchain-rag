import itertools
import sys
from typing import (
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
    no_type_check,
)

from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_rag.document_transformers.lazy_document_transformer import (
    LazyDocumentTransformer,
)

BATCH_SIZE = 100

T = TypeVar("T")  # Only in python 3.12
if sys.version_info.major > 3 or sys.version_info.minor >= 12:
    from itertools import batched  # type: ignore[attr-defined]
else:

    def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch


async def async_batched(
    async_iterator: AsyncIterator[T], n: int
) -> AsyncIterator[AsyncIterator[T]]:
    if n < 1:
        raise ValueError("n must be at least one")

    async def _async_iterator(iterable: Iterable[Any]) -> AsyncIterator[Any]:
        for i in iterable:
            yield i

    batch = []
    i = 0
    async for item in async_iterator:
        i += 1
        batch.append(item)
        if i == n:
            yield _async_iterator(batch)  # type: ignore
            i = 0
            batch.clear()
    if batch:
        yield _async_iterator(batch)  # type: ignore


class DocumentTransformers(LazyDocumentTransformer):
    """
    Document transformer that uses a list of Transformers.
    Take each input document, and apply all transformations present in the
    `transformers` attribute.

    This is the basis for multiple transformations, using the plus operator.
    """

    def __init__(
        self,
        transformers: Sequence[BaseDocumentTransformer],
        batch_size: int = BATCH_SIZE,
    ):
        self.transformers = transformers
        self.batch_size = batch_size

    def __add__(
        self,
        other: LazyDocumentTransformer,
    ) -> "DocumentTransformers":
        """Compose this runnable with another object to create a RunnableSequence."""
        if isinstance(other, DocumentTransformers):
            return DocumentTransformers(
                transformers=list(other.transformers) + list(self.transformers),
            )
        else:
            return DocumentTransformers(transformers=list(self.transformers) + [other])

    @no_type_check  # Bug in Mypy
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an iterator of documents with the list of transformations.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        # Can be refactored to use parallelism
        # Implementation when all transformers are NOT compatible with Runnable
        # It's not compatible with lazy strategy. Load all documents and apply
        # all transformations.
        for batch in batched(documents, self.batch_size):
            for t in self.transformers:
                for doc in t.transform_documents(documents=list(batch)):
                    yield doc

    @no_type_check  # Bug in Mypy
    async def _alazy_transform_documents(
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents with a list
        of transformations.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An iterator of transformed Documents.
        """
        # Implementation when all transformers are NOT compatible with Runnable
        # It's not compatible with lazy strategy. Load all documents and apply
        # all transformations.
        async for batch in async_batched(documents, self.batch_size):
            sync_batch = [doc async for doc in batch]
            for transformer in self.transformers:
                for doc in await transformer.atransform_documents(sync_batch):
                    yield doc
