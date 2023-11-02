import itertools
import sys
from functools import partial
from typing import Sequence, Any, Iterator, cast, Container

from langchain.schema import BaseDocumentTransformer, Document
from langchain.schema.runnable import RunnableParallel, RunnableGenerator

from .runnable_document_transformer import RunnableGeneratorDocumentTransformer


if sys.version_info.major > 3 or sys.version_info.minor >10:
    from itertools import batched
else:
    def batched(iterable, n):  # Only in python 3.12
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch


BATCH_SIZE = 16


def _transform_documents_generator(
        documents: Iterator[Document], *,
        transformers: Sequence[BaseDocumentTransformer]) -> \
        Container[Document]:
    steps = {f"transform_documents_{i}": transformer for i, transformer in
             enumerate(transformers)}
    for batch in batched(documents, BATCH_SIZE):
        result = RunnableParallel(steps=steps).invoke(batch)
        for chunk in result["steps"].values():
            yield list(chunk)
    # FIXME: a virer
    # all_results = []
    # for batch in batched(documents, BATCH_SIZE):
    #     result = RunnableParallel(steps=steps).invoke(batch)
    #     all_results.extend(result["steps"].values())
    # for chunk in all_results:
    #     yield list(chunk)

# TODO: voir opérateur & pour l'addition
class DocumentTransformers(RunnableGeneratorDocumentTransformer):
    """Document transformer that uses a list of Transformers."""

    class Config:
        arbitrary_types_allowed = True

    transformers: Sequence[BaseDocumentTransformer]
    """List of document transformer that are applied in parallel."""

    def lazy_transform_documents(
            self, documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an interator of documents with the list of transformations.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        # Version stream
        result = RunnableGenerator[Document, Document](
            transform=partial(_transform_documents_generator,
                              transformers=self.transformers)
        ).transform(documents)  # FIXME: invoke ?
        return cast(Iterator[Document], iter((y for chunk in result for y in chunk)))

    async def alazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        """Asynchronously transform an iterator of documents with a list of transformations.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An interator of transformed Documents.
        """
        # FIXME
