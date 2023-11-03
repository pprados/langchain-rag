from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Iterator, Optional, Sequence, Union, \
    Iterable, TypeVar

from langchain.schema import BaseDocumentTransformer, Document
from langchain.schema.runnable import RunnableConfig, RunnableSerializable

"""
    We propose an alternative way of making transformers compatible with LCEL.
    The first keeps the current protocol (RunnableDocumentTransformer).
    The second takes advantage of this to propose
    a lazy approach to transformations (RunnableGeneratorDocumentTransformer).
    It's better for the memory, pipeline, etc.
    
    Now, it's possible to create a pipeline of transformer like:
    Example:
    ..code - block:: python
    class UpperTransformer(RunnableGeneratorDocumentTransformer):
        def lazy_transform_documents(
                self,
                documents: Iterator[Document],
                **kwargs: Any
        ) -> Iterator[Document]:
            ...

        async def alazy_transform_documents(
                self,
                documents: Union[AsyncIterator[Document],Iterator[Document]],
                **kwargs: Any
        ) -> AsyncIterator[Document]:
            ...

    runnable = (UpperTransformer() | LowerTransformer())
    result = list(runnable.invoke(documents))
"""

T = TypeVar("T")


async def _to_async_iterator(iterator: Iterable[T]) -> AsyncIterator[T]:
    """Convert an iterable to an async iterator."""
    for item in iterator:
        yield item


class RunnableGeneratorDocumentTransformer(
    RunnableSerializable[
        Union[AsyncIterator[Document], Iterator[Document]],  # input
        Union[AsyncIterator[Document], Iterator[Document]]  # output
    ],
    BaseDocumentTransformer,
):
    """
    Runnable Document Transformer with lazy transformation.

    This class is a transition class for proposing lazy transformers, compatible with LCEL.
    Later, it can be integrated into BaseDocumentTransformer
    if you agree to add a lazy approach to transformations.
    All subclass of BaseDocumentTransformer must be updated to be compatible with this.
    """

    def transform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return list(self.lazy_transform_documents(iter(documents), **kwargs))

    async def atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return [
            doc
            async for doc in self.alazy_transform_documents(iter(documents), **kwargs)
        ]

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
        # return iter(self.transform_documents(list(documents)))
        raise NotImplementedError("not yet")

    @abstractmethod
    async def alazy_transform_documents(
            self, documents: Union[AsyncIterator[Document], Iterator[Document]],
            **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An interator of transformed Documents.
        """
        # Default implementation. Not realy lazy.
        # return iter(await self.atransform_documents(list(documents)))
        raise NotImplementedError("not yet")

    def invoke(
            self,
            input: Union[AsyncIterator[Document], Iterator[Document]],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Union[AsyncIterator[Document], Iterator[Document]]:
        if hasattr(input, "__aiter__"):
            raise ValueError("Use ainvoke")
        config = config or {}

        if hasattr(self, "lazy_transform_documents"):
            return self.lazy_transform_documents(input, **config)

        # Default implementation, without generator
        return iter(self.transform_documents(list(input), **config))

    async def ainvoke(
            self,
            input: Union[Iterable[Document], AsyncIterator[Document]],
            config: Optional[RunnableConfig] = None,
            **kwargs: Optional[Any],
    ) -> Union[AsyncIterator[Document], Iterator[Document]]:
        # # Default implementation, without generator
        config = config or {}
        return self.alazy_transform_documents(documents=input, **config)


class RunnableDocumentTransformer(
    RunnableSerializable[Sequence[Document], Sequence[Document]],
    BaseDocumentTransformer,
):
    """
    Runnable Document Transformer with lazy transformation.
    This class is a transition class for proposing lazy transformers, compatible with LCEL.
    Later, it can be integrated into BaseDocumentTransformer
    if you refuse to add a lazy approach to transformations.
    All subclass of BaseDocumentTransformer must be updated to be compatible with this.
    """
    """  # FIXME
    Now, it's possible to create a chain of transformations
    (only if the transformation is compatible with `RunnableDocumentTransformer`)
    """

    def invoke(
            self, input: Sequence[Document], config: Optional[RunnableConfig] = None
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
        return await self.atransform_documents(input, **config)
