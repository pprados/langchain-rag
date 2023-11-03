import asyncio
import copy
from collections.abc import AsyncIterator
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, cast, Iterator

from langchain.chains import LLMChain
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel

from langchain_rag.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
)


def _default_get_input(doc: Document) -> Dict[str, Any]:
    """Return the context chain input."""
    return {
        "context": doc.page_content,
    }


_default_template = """
Given a text input, generate {nb_of_questions} questions from it in the same language. 
Context:
```
{context}
```
{format_instructions}"""

_default_parser = NumberedListOutputParser()


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        template=_default_template,
        output_parser=_default_parser,
        partial_variables={
            "format_instructions": _default_parser.get_format_instructions()
        },
    )


class GenerateQuestionsTransformer(RunnableGeneratorDocumentTransformer):
    """Generate questions for each Documents."""

    llm_chain: LLMChain
    get_input: Callable[[Document], dict] = _default_get_input
    nb_of_questions: int = 3

    """Callable for constructing the chain input from the query and a Document."""

    def lazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        """Compress page content of raw documents."""
        _callbacks = kwargs.get("callbacks", None)
        for doc in documents:
            _input = {
                **self.get_input(doc),
                **{"nb_of_questions": self.nb_of_questions},
            }
            output = cast(
                Sequence[str], self.llm_chain.predict(callbacks=_callbacks, **_input)
            )
            if not output:
                continue
            for question in output:
                yield Document(page_content=question, metadata=doc.metadata)

    def transform_documents(
            self,
            documents: Sequence[Document],
            **kwargs: Any
    ) -> Sequence[Document]:
        return list(self.lazy_transform_documents(documents=iter(documents), **kwargs))

    async def lazy_atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> AsyncIterator[Document]:
        """Compress page content of raw documents asynchronously."""
        _callbacks = kwargs.get("callbacks", None)
        outputs = await asyncio.gather(
            *[
                self.llm_chain.apredict(
                    **self.get_input(documents), callbacks=_callbacks
                )
                for doc in documents
            ]
        )
        for i, doc in enumerate(documents):
            if not outputs[i]:
                continue
            metadata = copy.deepcopy(doc.metadata)
            metadata["transformer"] = self.__class__.__name__
            yield Document(page_content=outputs[i], metadata=metadata)

    async def atransform_documents(
            self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
        # FIXME: a tester. Lazy ?
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.transform_documents, **kwargs), documents
        )

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: Optional[PromptTemplate] = None,
            get_input: Optional[Callable[[Document], dict]] = None,
            nb_of_questions: int = 3,
            llm_chain_kwargs: Optional[dict] = None,
    ) -> "GenerateQuestionsTransformer":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else _default_get_input
        llm_chain = LLMChain(
            llm=llm,
            prompt=_prompt,
            output_parser=_prompt.output_parser,
            **(llm_chain_kwargs or {}),
        )
        return cls(
            llm_chain=llm_chain, get_input=_get_input, nb_of_questions=nb_of_questions
        )
