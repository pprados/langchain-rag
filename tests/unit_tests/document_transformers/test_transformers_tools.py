from typing import Any, Dict, List, Mapping, Optional, cast

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.documents import Document

# Note: Import directly from langchain_core is not stable and generate some errors
from langchain_core.language_models import LLM, BaseLLM
from langchain_core.pydantic_v1 import validator

from langchain_rag.document_transformers import (
    DocumentTransformerPipeline,
    DocumentTransformers,
)
from langchain_rag.document_transformers.generate_questions import (
    GenerateQuestionsTransformer,
)
from langchain_rag.document_transformers.summarize_and_questions_transformer import (
    SummarizeAndQuestionsTransformer,
)
from langchain_rag.document_transformers.summarize_transformer import (
    SummarizeTransformer,
)
from langchain_rag.document_transformers.tfidf_transformer import \
    LemmatizeDocumentTransformer, StemmerDocumentTransformer, TFIDFTransformer
from tests.unit_tests.documents.sample_transformer import (
    LowerLazyTransformer,
    UpperLazyTransformer,
)

TEMPERATURE = 0.0
MAX_TOKENS = 1000
FAKE_LLM = True
USE_CACHE = True


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @validator("queries", always=True)
    def check_queries_required(
        cls, queries: Optional[Mapping], values: Mapping[str, Any]
    ) -> Optional[Mapping]:
        if values.get("sequential_response") and not queries:
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return queries

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response


def init_llm(
    queries: Dict[int, str],
    max_token: int = MAX_TOKENS,
) -> BaseLLM:
    if FAKE_LLM:
        return FakeLLM(
            queries=queries,
            sequential_responses=True,
        )
    else:
        import langchain
        from dotenv import load_dotenv
        from langchain_community.cache import SQLiteCache

        load_dotenv()

        if USE_CACHE:
            langchain.llm_cache = SQLiteCache(
                database_path="/tmp/cache_qa_with_reference.db"
            )
        llm = langchain.OpenAI(
            temperature=TEMPERATURE,
            max_tokens=max_token,
            # cache=False,
        )
        return llm


# %% generate_questions
def test_generate_questions_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used "
            "in the past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 6


def test_generate_questions_lazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, 
    formulas and related structures, shapes and the spaces in which they are 
    contained, and quantities and their changes. These topics are represented 
    in modern mathematics with the major subdisciplines of number theory, algebra, 
    geometry, and analysis, respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used in the "
            "past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 6


@pytest.mark.asyncio
async def test_generate_questions_atransform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation "
            "used in the past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 6


@pytest.mark.asyncio
async def test_generate_questions_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, and 
    quantities and their changes. These topics are represented in modern mathematics 
    with the major subdisciplines of number theory, algebra, geometry, and analysis, 
    respectively. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "\n\n"
            "1. What are the major subdisciplines of mathematics?\n"
            "2. What topics are represented in modern mathematics?\n"
            "3. What are the topics of mathematics that include numbers, "
            "formulas and related structures, shapes and the spaces in "
            "which they are contained, and quantities and their changes?\n",
            1: "\n\n"
            "1. What are the origins of mathematics?\n"
            "2. What are some of the mathematical methods and notation used in the "
            "past?\n"
            "3. How has the history of mathematics evolved over time?\n",
        }
    )
    transformer = GenerateQuestionsTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 6


# %% sumarize_transformer


def test_sumarize_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, shapes, "
            "spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries and the "
            "development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 2


def test_sumarize_transformer_lazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, "
            "shapes, spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries and "
            "the development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 2


@pytest.mark.asyncio
async def test_sumarize_transformer_atransform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, shapes, "
            "spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries and "
            "the development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 2


@pytest.mark.asyncio
async def test_sumarize_transformer_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "SUMMARY:\nMathematics is the study of numbers, formulas, "
            "shapes, spaces, "
            "quantities, and their changes.",
            1: "SUMMARY:\nMathematics has a long history of discoveries "
            "and the development "
            "of mathematical methods and notation.",
        }
    )
    transformer = SummarizeTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 2


# %% sumarize_and_questions_transformer


def test_sumarize_and_questions_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, and their "
            'changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin of '
            "discoveries in mathematics and the mathematical methods and notation "
            'of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 8


def test_sumarize_and_questions_transformer_lazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, and their "
            'changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin '
            "of discoveries in mathematics and the mathematical methods and "
            'notation of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = list(transformer.lazy_transform_documents(iter([doc1, doc2])))
    assert len(result) == 8


@pytest.mark.asyncio
async def test_sumarize_and_questions_transformer_atransform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, and their "
            'changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin of '
            "discoveries in mathematics and the mathematical methods and notation "
            'of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = await transformer.atransform_documents([doc1, doc2])
    assert len(result) == 8


@pytest.mark.asyncio
async def test_sumarize_and_questions_transformer_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    llm = init_llm(
        {
            0: "Output:\n"
            "```\n"
            "{\n"
            '    "summary": "Mathematics is an area of knowledge that includes topics '
            "such as numbers, formulas, shapes, spaces, quantities, "
            'and their changes.",\n'
            '    "questions": [\n'
            '        "What topics are included in mathematics?",\n'
            '        "What are some examples of mathematical structures?",\n'
            '        "How do quantities and their changes relate to mathematics?"\n'
            "    ]\n"
            "}\n"
            "```\n",
            1: "Output:\n"
            "    ```\n"
            "    {\n"
            '        "summary": "The history of mathematics deals with the origin '
            "of discoveries in mathematics and the mathematical methods and "
            'notation of the past.",\n'
            '        "questions": [\n'
            '            "What is the history of mathematics?",\n'
            '            "What discoveries have been made in mathematics?",\n'
            '           "What are the mathematical methods and notation of the past?"\n'
            "        ]\n"
            "    }\n"
            "    ```\n",
        }
    )
    transformer = SummarizeAndQuestionsTransformer.from_llm(llm=llm)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 8


# %%
@pytest.mark.asyncio
async def test_DocumentTransformerPipeline_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    trans1 = LowerLazyTransformer()
    trans2 = UpperLazyTransformer()
    transformer = DocumentTransformerPipeline(
        transformers=[trans1, trans2], batch_size=1
    )
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 2
    assert result[0].page_content == doc1.page_content.lower()


@pytest.mark.asyncio
async def test_DocumentTransformers_alazy_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    trans1 = LowerLazyTransformer()
    trans2 = UpperLazyTransformer()
    transformer = DocumentTransformers(transformers=[trans1, trans2], batch_size=100)
    result = [
        doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))
    ]
    assert len(result) == 4
    assert result[0].page_content == doc1.page_content.lower()
    assert result[2].page_content == doc1.page_content.upper()


def test_lematize_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    transformer = LemmatizeDocumentTransformer(language="english")
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 2
    assert (result[0].page_content ==
            'mathematics area knowledge includes topic number formula related '
            'structure shape space contained quantity change')
    assert (result[1].page_content ==
            'history mathematics deal origin discovery mathematics mathematical '
            'method notation past')


def test_stemmer_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    transformer = StemmerDocumentTransformer(language="english")
    result = transformer.transform_documents([doc1, doc2])
    assert len(result) == 2
    assert (result[0].page_content ==
            'mathemat area knowledg includ topic number formula relat structur shape '
            'space contain quantiti chang')
    assert (result[1].page_content ==
            'histori mathemat deal origin discoveri mathemat mathemat method '
            'notat past')


def test_tfid_transformer_transform_documents() -> None:
    doc1 = Document(
        page_content="""
    Mathematics is an area of knowledge that includes the topics of numbers, formulas 
    and related structures, shapes and the spaces in which they are contained, 
    and quantities and their changes. 
    """
    )
    doc2 = Document(
        page_content="""
    The history of mathematics deals with the origin of discoveries in mathematics and 
    the mathematical methods and notation of the past.'
    """
    )
    transformer = TFIDFTransformer()
    result = transformer.transform_documents([doc1, doc2])
    assert transformer.tfidf_retriever != None
    assert len(result) == 0


