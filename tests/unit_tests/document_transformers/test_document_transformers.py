from typing import Sequence

import pytest
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

from langchain_rag.document_transformers.document_transformers import (
    _LEGACY,
    DocumentTransformers,
)
from tests.unit_tests.document_transformers.sample_transformer import (
    LowerLazyTransformer,
)
from tests.unit_tests.document_transformers.test_runnable_transformers import (
    UpperLazyTransformer,
)


def by_pg(doc: Document) -> str:
    return doc.page_content


@pytest.mark.skipif(not _LEGACY, reason="Test only runnable transformer")
@pytest.mark.parametrize(
    "transformers",
    [[TokenTextSplitter(chunk_size=1, chunk_overlap=0), UpperLazyTransformer()]],
)
def test_document_transformers_legacy(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")

    transfomer = DocumentTransformers(transformers=transformers)
    r = transfomer.transform_documents([doc1, doc2])
    assert len(r) == 6
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content="my"),
            Document(page_content=" test"),
            Document(page_content=doc1.page_content.upper()),
            Document(page_content="other"),
            Document(page_content=" test"),
            Document(page_content=doc2.page_content.upper()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
def test_transform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transfomer = DocumentTransformers(transformers=transformers)
    r = transfomer.transform_documents([doc1, doc2])
    assert len(r) == 4
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
@pytest.mark.asyncio
async def test_atransform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transfomer = DocumentTransformers(transformers=transformers)
    r = await transfomer.atransform_documents([doc1, doc2])
    assert len(r) == 4
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
def test_lazy_transform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transfomer = DocumentTransformers(transformers=transformers)
    r = [doc for doc in transfomer.lazy_transform_documents(iter([doc1, doc2]))]
    assert len(r) == 4

    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
@pytest.mark.asyncio
async def test_alazy_transform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transfomer = DocumentTransformers(transformers=transformers)
    r = [doc async for doc in transfomer.alazy_transform_documents(iter([doc1, doc2]))]
    assert len(r) == 4
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )
