from typing import Sequence, Optional, Any, Iterator, AsyncIterator

from langchain.schema import Document, BaseDocumentTransformer
from langchain.schema.runnable import RunnableSerializable, RunnableConfig
from .document_transformer import \
    GeneratorBaseDocumentTransformer


class RunnableDocumentTransformer(
    RunnableSerializable[Sequence[Document], Sequence[Document]],
    BaseDocumentTransformer):
    """ This is a transition class. It must be integrated in BaseDocumentTransformer.
    Transform a `BaseDocumentTransformer`to a `RunnableDocumentTransformer`.
    Now, it's possible to create a chain of transformations
    (only if the transformation is compatible with `RunnableDocumentTransformer`)
    """

    def invoke(
            self,
            input: Sequence[Document],
            config: Optional[RunnableConfig] = None
    ) -> Sequence[Document]:
        config = config or {}
        return self.transform_documents(
            input,
            **config,
        )

    async def ainvoke(
            self,
            input: Sequence[Document],
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Sequence[Document]:
        config = config or {}
        return await self.atransform_documents(
            input,
            **config
        )


class RunnableGeneratorDocumentTransformer(
    RunnableSerializable[Sequence[Document], Sequence[Document]],
    GeneratorBaseDocumentTransformer):

    def invoke(
            self,
            input: Iterator[Document],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> Iterator[Document]:
        config = config or {}

        if hasattr(self, "lazy_transform_documents"):
            return self.lazy_transform_documents(input,
                                                 **config)

        # Default implementation, without generator
        return iter(self.transform_documents(
            list(input),
            **config
        ))

    async def ainvoke(
            self,
            input: Iterator[Document],  # TODO: accept |Sequence[Document] ?
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> AsyncIterator[Document]:
        # Default implementation, without generator
        config = config or {}
        if hasattr(self, "alazy_transform_documents"):
            return self.alazy_transform_documents(input,**config)
        # return (for x in await self.atransform_documents(
        #     list(input),
        #     **config
        # ))
        # FIXME