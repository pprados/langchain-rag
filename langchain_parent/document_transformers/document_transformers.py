from functools import partial
from itertools import chain
from typing import Sequence, Any, Iterator
from functools import partial

from langchain.schema import BaseDocumentTransformer, Document
from langchain.schema.runnable import RunnableParallel, RunnableGenerator

from .runnable_document_transformer import RunnableGeneratorDocumentTransformer


# FIXME: dans le template Pydantic, il faut ajouter qu'une liste avec , à la fin est mal formée.


def _transform_documents_generator(documents: Iterator[Document],*,transformers:Sequence[BaseDocumentTransformer]) -> Iterator[Document]:
    steps = {f"step_{i}": transformer for i, transformer in
             enumerate(transformers)}
    result = RunnableParallel(steps=steps
                              ).invoke(documents)
    for doc in chain(result['steps'].values()):
        yield from doc


class DocumentTransformers(RunnableGeneratorDocumentTransformer):
    """Document transformer that uses a pipeline of Transformers."""

    class Config:
        arbitrary_types_allowed=True

    transformers: Sequence[BaseDocumentTransformer]

    """List of document transformer that are applied in sequence."""
    def lazy_transform_documents(
            self, documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        """Transform a list of documents."""

        return RunnableGenerator(
            transform=partial(_transform_documents_generator,
                              transformers=self.transformers)
        ).transform(documents)

    async def alazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        """Compress retrieved documents given the query context."""
        # FIXME
