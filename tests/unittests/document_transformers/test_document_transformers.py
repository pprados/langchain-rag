from typing import Type, Tuple

import pytest
from langchain.schema import Document

from langchain_rag.document_transformers.document_transformers import \
    DocumentTransformers, _COMPATIBLE_RUNNABLE
from tests.unittests.document_transformers.sample_transformer import LowerTransformer, \
    LowerLazyTransformer
from tests.unittests.document_transformers.test_runnable_transformers import \
    UpperLazyTransformer, UpperTransformer


@pytest.mark.skipif(_COMPATIBLE_RUNNABLE,
                    reason="Test only runnable transformer")
@pytest.mark.parametrize("cls", [UpperLazyTransformer, UpperTransformer])
def test_document_transformers(cls: Type) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
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
        Document(page_content="my"),
        Document(page_content=" test"),
        Document(page_content="other"),
        Document(page_content=" test"),
        Document(page_content=doc1.page_content.upper()),
        Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.skipif(not _COMPATIBLE_RUNNABLE,
                    reason="Test *future* runnable transformation")
@pytest.mark.parametrize("cls", [(UpperLazyTransformer, LowerLazyTransformer),
                                 (UpperTransformer, LowerTransformer)])
def test_document_transformers_runnable(cls: Tuple[Type, Type]) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transfomer = DocumentTransformers(
        transformers=[
            cls[0](),
            cls[1](),
        ]
    )
    r = transfomer.transform_documents([doc1, doc2])
    assert len(r) == 4
    assert r == [
        Document(page_content=doc1.page_content.upper()),
        Document(page_content=doc2.page_content.upper()),
        Document(page_content=doc1.page_content.lower()),
        Document(page_content=doc2.page_content.lower()),
    ]
