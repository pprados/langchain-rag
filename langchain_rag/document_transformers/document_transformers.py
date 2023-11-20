import itertools
import sys
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
    Union, AsyncIterable,
)

from langchain.schema import BaseDocumentTransformer, Document
from langchain.schema.runnable import RunnableGenerator, RunnableParallel

from .runnable_document_transformer import RunnableGeneratorDocumentTransformer, \
    to_async_iterator

if sys.version_info.major > 3 or sys.version_info.minor > 10:
    from itertools import batched  # type: ignore[attr-defined]
else:
    T = TypeVar("T")  # Only in python 3.12

    def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

BATCH_SIZE = 1 # FIXME: 16
# The Runnable interface is compatible runnable?
_COMPATIBLE_RUNNABLE = True


def _transform_documents_generator(
    documents: Iterator[Document],
    *,
    transformers: Sequence[RunnableGeneratorDocumentTransformer],
) -> Iterator[Document]:
    Input = Union[AsyncIterator[Document], Iterator[Document]]
    steps = {
        f"transform_documents_{i}": transformer
        for i, transformer in enumerate(transformers)
    }
    # Implementation when all transformers are compatible with Runnable
    for batch in batched(documents, BATCH_SIZE):
        result = RunnableParallel[Input](steps=steps).invoke(batch)
        for chunk in result["steps"].values():
            yield chunk

async def abatched(iterable: AsyncIterable[T], n: int) -> AsyncIterable[Tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    batch=[]
    async for t in iterable:
        batch.append(t)
        if len(batch)> n:
            yield batch
            batch.clear()
    if batch:
        yield batch


async def _atransform_documents_generator(
    documents: AsyncIterator[Document],
    *,
    transformers: Sequence[RunnableGeneratorDocumentTransformer],
) -> AsyncIterator[Document]:
    Input = AsyncIterator[Document]
    steps = {
        f"transform_documents_{i}": transformer
        for i, transformer in enumerate(transformers)
    }
    # Implementation when all transformers are compatible with Runnable
    batch=[]
    async for doc in documents:
        batch.append(doc)
        if len(batch)> BATCH_SIZE:
            yield batch
            batch.clear()
    if batch:
        yield batch

    async for batch in abatched(documents, BATCH_SIZE):
        result = await RunnableParallel[Input](steps=steps).ainvoke(batch)
        for chunk in result["steps"].values():
            yield chunk


class DocumentTransformers(RunnableGeneratorDocumentTransformer):
    """Document transformer that uses a list of Transformers."""

    class Config:
        arbitrary_types_allowed = True

    if _COMPATIBLE_RUNNABLE:
        transformers: Sequence[RunnableGeneratorDocumentTransformer]
    else:
        transformers: Sequence[BaseDocumentTransformer]
    """List of document transformer that are applied in parallel."""

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an interator of documents with the list of transformations.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        if _COMPATIBLE_RUNNABLE:
            for batch in batched(documents, BATCH_SIZE):
                for t in self.transformers:
                    for doc in t.lazy_transform_documents(iter(batch)):
                        yield doc
        else:
            # Implementation when all transformers are NOT compatible with Runnable
            # It's not compatible with lazy strategy. Load all documents and apply
            # all transformations.
            docs = [doc for doc in documents]
            for transformer in self.transformers:
                for doc in transformer.transform_documents(documents=docs):
                    yield doc

    async def _alazy_transform_documents(
        self,
        documents: AsyncIterator[Document],
        **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents with a list
        of transformations.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An interator of transformed Documents.
        """
        if _COMPATIBLE_RUNNABLE:

            # # Get a batch of documents, then apply each transformation by batch
            async for batch in abatched(documents, BATCH_SIZE):
                for t in self.transformers:
                    async for doc in t.alazy_transform_documents(iter(batch)):
                        yield doc

        else:
            # Implementation when all transformers are NOT compatible with Runnable
            # It's not compatible with lazy strategy. Load all documents and apply
            # all transformations.
            docs = [doc async for doc in documents]
            for transformer in self.transformers:
                for doc in await transformer.atransform_documents(documents=docs):
                    yield doc
