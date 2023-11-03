import asyncio
import copy
from typing import Iterator, Any, Sequence, AsyncIterator, Union, Type

import pytest


from langchain.schema import Document

from langchain_rag.document_transformers import RunnableDocumentTransformer
from langchain_rag.document_transformers.runnable_document_transformer import \
    RunnableGeneratorDocumentTransformer, _to_async_iterator


class UpperLazyTransformer(RunnableGeneratorDocumentTransformer):
    """ Implementation of a runnable transformer, with lazy transformation """
    def lazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        return (Document(page_content=doc.page_content.upper(),
                             metadata=copy.deepcopy(doc.metadata))
                    for doc in documents)

    async def alazy_transform_documents(
            self,
            documents: Union[AsyncIterator[Document],Iterator[Document]],
            **kwargs: Any
    ) -> AsyncIterator[Document]:

        if hasattr(documents, "__aiter__"):
            async_documents = documents  # type: ignore[assignment]
        else:
            async_documents = _to_async_iterator(documents)

        async for doc in async_documents:
            yield Document(
                page_content=doc.page_content.upper(),
                metadata=copy.deepcopy(doc.metadata))

class UpperTransformer(RunnableDocumentTransformer):
    """ Implementation of a runnable transformer, without lazy transformation """
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return [Document(page_content=doc.page_content.upper(),
                             metadata=copy.deepcopy(doc.metadata))
                    for doc in documents]

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        return self.transform_documents(documents=documents,**kwargs)


@pytest.mark.parametrize("cls", [UpperLazyTransformer,UpperTransformer])
def test_transform(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = cls().transform_documents(documents=[doc1, doc2])
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


@pytest.mark.parametrize("cls", [UpperLazyTransformer,UpperTransformer])
@pytest.mark.asyncio
async def test_atransform(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = await cls().atransform_documents(documents=[doc1, doc2])
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
def test_lazy_transform(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = list(
        cls().lazy_transform_documents(documents=iter([doc1, doc2])))
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_alazy_transform_sync_iterator(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = [doc async for doc in cls().alazy_transform_documents(
        documents=iter([doc1, doc2]))]
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]

@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_alazy_transform_async_iterator(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = [doc async for doc in cls().alazy_transform_documents(
        documents=_to_async_iterator(iter([doc1, doc2])))]
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


@pytest.mark.parametrize("cls", [UpperLazyTransformer,UpperTransformer])
def test_invoke_transformer(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = list(cls().invoke(iter([doc1, doc2])))
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_ainvoke_transformer_sync_iterator(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = [doc async for doc in await cls().ainvoke(iter([doc1, doc2]))]
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]

@pytest.mark.parametrize("cls", [UpperTransformer])
@pytest.mark.asyncio
async def test_ainvoke_transformer(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = await cls().ainvoke([doc1, doc2])
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]

@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_ainvoke_transformer_async_iterator(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = [doc async for doc in await cls().ainvoke(
        _to_async_iterator(iter([doc1, doc2])))]
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]

@pytest.mark.parametrize("cls", [UpperTransformer])
@pytest.mark.asyncio
async def test_ainvoke_transformer(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    r = await cls().ainvoke([doc1, doc2])
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


# %%
@pytest.mark.parametrize("cls", [UpperLazyTransformer,UpperTransformer])
def test_invoke_pipeline(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    runnable = (cls() | cls())  # LCEL syntax
    r= list(runnable.invoke([doc1,doc2]))
    assert r == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]

@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_ainvoke_pipeline_sync_iterator(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    runnable = (cls() | cls())
    r= await runnable.ainvoke(_to_async_iterator(iter([doc1,doc2])))
    rr=[doc async for doc in r]
    assert rr == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]

@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_ainvoke_pipeline_async_iterator(cls:Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    runnable = (cls() | cls())
    r= await runnable.ainvoke(iter([doc1,doc2]))
    rr=[doc async for doc in r]
    assert rr == [Document(page_content=doc1.page_content.upper()),
                 Document(page_content=doc2.page_content.upper())]


@pytest.mark.parametrize("cls", [UpperTransformer])
@pytest.mark.asyncio
async def test_ainvoke_pipeline(cls: Type):
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    runnable = (cls() | cls())
    r = await runnable.ainvoke(iter([doc1, doc2]))
    assert r == [Document(page_content=doc1.page_content.upper()),
                  Document(page_content=doc2.page_content.upper())]
