import asyncio
import threading
import time
from abc import abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Iterator,
    Sequence,
    TypeVar,
    Union,
)

from langchain_core.documents import BaseDocumentTransformer, Document

"""
    We propose an alternative way of making transformers compatible with LCEL.
    The first keeps the current protocol (RunnableDocumentTransformer).
    The second takes advantage of this to propose
    a lazy approach to transformations (_RunnableGeneratorDocumentTransformer).
    It's better for the memory, pipeline, etc.

    Now, it's possible to create a pipeline of transformer like:
    Example:
    ..code - block:: python
    class UpperTransformer(_RunnableGeneratorDocumentTransformer):
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


async def to_async_iterator(iterator: Iterator[T]) -> AsyncIterator[T]:
    """Convert an iterable to an async iterator."""
    for item in iterator:
        yield item


_DONE = ""
_TIMEOUT = 1


def to_sync_iterator(async_iterable: AsyncIterator[T], maxsize: int = 0) -> Iterator[T]:
    def _run_coroutine(
        loop: asyncio.AbstractEventLoop,
        async_iterable: AsyncIterator[T],
        queue: asyncio.Queue,
    ) -> None:
        async def _consume_async_iterable(
            async_iterable: AsyncIterator[T], queue: asyncio.Queue
        ) -> None:
            async for x in async_iterable:
                await queue.put(x)

            await queue.put(_DONE)

        loop.run_until_complete(_consume_async_iterable(async_iterable, queue))

    queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
    loop = asyncio.new_event_loop()

    t = threading.Thread(target=_run_coroutine, args=(loop, async_iterable, queue))
    t.daemon = True
    t.start()

    while True:
        if not queue.empty():
            x = queue.get_nowait()

            if x is _DONE:
                break
            else:
                yield x
        else:
            time.sleep(_TIMEOUT)

    t.join()


Input = Union[AsyncIterator[Document], Iterator[Document], Sequence[Document]]
Output = Union[AsyncIterator[Document], Iterator[Document]]


class LazyDocumentTransformer(BaseDocumentTransformer):
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        # Convert lazy to classical transformation
        return list(self.lazy_transform_documents(iter(documents), **kwargs))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return [
            doc
            async for doc in self.alazy_transform_documents(iter(documents), **kwargs)
        ]

    @staticmethod
    def _doc_to_async_iterator(
        documents: Sequence[Document] | AsyncIterator[Document] | Iterator[Document],
    ) -> AsyncIterator[Document]:
        if isinstance(documents, Sequence):
            async_documents = to_async_iterator(iter(documents))
        elif isinstance(documents, AsyncIterator):
            async_documents = documents
        elif isinstance(documents, Iterator):
            async_documents = to_async_iterator(documents)
        else:
            raise ValueError("Invalid input type")
        return async_documents

    @abstractmethod
    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        """Transform an iterator of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            An iterator oftransformed Documents.
        """
        raise NotImplementedError()

    @abstractmethod
    async def _alazy_transform_documents(  # type: ignore
        self, documents: AsyncIterator[Document], **kwargs: Any
    ) -> AsyncGenerator[Document, None]:
        yield None  # type: ignore

    async def alazy_transform_documents(
        self,
        documents: Input,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> AsyncIterator[Document]:
        """Asynchronously transform an iterator of documents.

        Args:
            documents: An iterator of Documents to be transformed.

        Returns:
            An iterator of transformed Documents.
        """
        async_documents = LazyDocumentTransformer._doc_to_async_iterator(documents)

        async for doc in self._alazy_transform_documents(async_documents):
            yield doc
