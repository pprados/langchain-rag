import copy
from typing import Iterator, Any, AsyncIterator, Union, Sequence, Callable

from langchain.schema import Document

from langchain_rag.document_transformers.runnable_document_transformer import \
    RunnableGeneratorDocumentTransformer, to_async_iterator, RunnableDocumentTransformer


class _LazyTransformer(RunnableGeneratorDocumentTransformer):
    """ Implementation of a runnable transformer, with lazy transformation """

    fn: Callable[[Any], str]

    def lazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        return (Document(page_content=self.fn(doc.page_content),
                         metadata=copy.deepcopy(doc.metadata))
                for doc in documents)

    async def alazy_transform_documents(  # type:ignore
            self,
            documents: Union[AsyncIterator[Document], Iterator[Document]],
            **kwargs: Any
    ) -> AsyncIterator[Document]:

        if isinstance(documents, AsyncIterator):
            async_documents = documents
        else:
            async_documents = to_async_iterator(documents)

        async for doc in async_documents:
            yield Document(
                page_content=self.fn(doc.page_content),
                metadata=copy.deepcopy(doc.metadata))


class _Transformer(RunnableDocumentTransformer):
    """ Implementation of a runnable transformer, without lazy transformation """

    fn: Callable[[Any], str]

    def transform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return [Document(page_content=self.fn(doc.page_content),
                         metadata=copy.deepcopy(doc.metadata))
                for doc in documents]

    async def atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return self.transform_documents(documents=documents, **kwargs)


class UpperLazyTransformer(_LazyTransformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.upper, **kwargs)


class UpperTransformer(_Transformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.upper, **kwargs)


class LowerLazyTransformer(_LazyTransformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.lower, **kwargs)


class LowerTransformer(_Transformer):
    def __init__(self, **kwargs: Any):
        super().__init__(fn=str.lower, **kwargs)

