import itertools
import sys
from functools import partial
from typing import Any, Iterator, Sequence, Iterable, TypeVar, Tuple, \
    Union, AsyncIterator, Mapping

from langchain.schema import BaseDocumentTransformer, Document
from langchain.schema.runnable import RunnableGenerator, RunnableParallel, Runnable

from .runnable_document_transformer import RunnableGeneratorDocumentTransformer

if sys.version_info.major > 3 or sys.version_info.minor > 10:
    from itertools import batched  # type: ignore[attr-defined]
else:

    T = TypeVar('T')  # Only in python 3.12


    def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

BATCH_SIZE = 16
_COMPATIBLE_RUNNABLE = True


def _transform_documents_generator(
        documents: Iterator[Document], *,
        transformers: Sequence[RunnableGeneratorDocumentTransformer]
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


class DocumentTransformers(RunnableGeneratorDocumentTransformer):
    """Document transformer that uses a list of Transformers."""

    class Config:
        arbitrary_types_allowed = True

    transformers: Sequence[BaseDocumentTransformer]
    # transformers: Sequence[RunnableGeneratorDocumentTransformer]  # FIXME: temporaire, pour tester
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
            # Version stream
            result = RunnableGenerator[Document, Document](
                transform=partial(
                    _transform_documents_generator, transformers=self.transformers
                )
            ).transform(
                input=documents,
                **kwargs,
            )
            yield from (y for chunk in result for y in chunk)  # type: ignore
        else:
            # Implementation when all transformers are NOT compatible with Runnable
            # It's not compatible with lazy strategy. Load all documents and apply
            # all transformations.
            docs = list(documents)
            for transformer in self.transformers:
                for doc in transformer.transform_documents(documents=docs):
                    yield doc

    async def alazy_transform_documents(
            self, documents: Union[AsyncIterator[Document], Iterator[Document]],
            **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents with a list of transformations.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An interator of transformed Documents.
        """
        raise NotImplementedError("Not yet")
