from typing import Tuple, Type

import pytest
import langchain
from langchain_rag.document_transformers.document_transformers import (
    _COMPATIBLE_RUNNABLE,
    DocumentTransformers,
)
from tests.unit_tests.document_transformers.sample_transformer import (
    LowerLazyTransformer,
    LowerTransformer,
)
from tests.unit_tests.document_transformers.test_runnable_transformers import (
    UpperLazyTransformer,
    UpperTransformer,
)


@pytest.mark.skipif(_COMPATIBLE_RUNNABLE, reason="Test only runnable transformer")
@pytest.mark.parametrize("cls", [UpperLazyTransformer, UpperTransformer])
def test_document_transformers(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    from langchain.text_splitter import TokenTextSplitter

    transfomer = DocumentTransformers(
        transformers=[
            TokenTextSplitter(chunk_size=1, chunk_overlap=0),
            cls(),
        ]
    )
    r = transfomer.transform_documents([doc1, doc2])
    assert len(r) == 6
    assert r == [
        langchain.schema.Document(page_content="my"),
        langchain.schema.Document(page_content=" test"),
        langchain.schema.Document(page_content="other"),
        langchain.schema.Document(page_content=" test"),
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.skipif(
    not _COMPATIBLE_RUNNABLE, reason="Test *future* runnable transformation"
)
@pytest.mark.parametrize(
    "cls",
    [
        (UpperLazyTransformer, LowerLazyTransformer),
    ],
)
def test_document_transformers_runnable(cls: Tuple[Type, Type]) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    transfomer = DocumentTransformers(
        transformers=[
            cls[0](),
            cls[1](),
        ]
    )
    r = transfomer.transform_documents([doc1, doc2])
    assert len(r) == 4
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc1.page_content.lower()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.lower()),
    ]


@pytest.mark.skipif(
    not _COMPATIBLE_RUNNABLE, reason="Test *future* runnable transformation"
)
@pytest.mark.parametrize(
    "cls",
    [
        (UpperLazyTransformer, LowerLazyTransformer),
        # (UpperTransformer, LowerTransformer),
    ],
)
@pytest.mark.asyncio
async def test_adocument_transformers_runnable(cls: Tuple[Type, Type]) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    transfomer = DocumentTransformers(
        transformers=[
            cls[0](),
            cls[1](),
        ]
    )
    r = await transfomer.atransform_documents([doc1, doc2])
    assert len(r) == 4
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
        langchain.schema.Document(page_content=doc1.page_content.lower()),
        langchain.schema.Document(page_content=doc2.page_content.lower()),
    ]
