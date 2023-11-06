import copy
from collections.abc import AsyncIterator
from typing import Any, Callable, Dict, Iterator, Optional, Union, cast

from langchain.chains import LLMChain
from langchain.output_parsers import NumberedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel

from langchain_rag.document_transformers.runnable_document_transformer import (
    RunnableGeneratorDocumentTransformer,
    to_async_iterator,
)


def _default_get_input(doc: Document) -> Dict[str, Any]:
    """Return the context chain input."""
    return {
        "context": doc.page_content,
    }


_default_template = """
Sumarize a text input in the same language. 
Context:
```
{context}
```
"""
_default_format_instruction = NumberedListOutputParser()


def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        template=_default_template,
    )


class SummarizeTransformer(RunnableGeneratorDocumentTransformer):
    """Generate questions for each Documents."""

    llm_chain: LLMChain
    get_input: Callable[[Document], dict] = _default_get_input

    """LLM wrapper to use for compressing documents."""

    """Callable for constructing the chain input from the query and a Document."""

    def lazy_transform_documents(
        self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        _callbacks = kwargs.get("callbacks", None)
        for doc in documents:
            _input = self.get_input(doc)
            output = self.llm_chain.predict(callbacks=_callbacks, **_input)
            if not output:
                continue
            metadata = copy.deepcopy(doc.metadata)
            metadata["transformer"] = self.__class__.__name__
            yield Document(
                page_content="SUMMARY:\n" + str(output).strip(), metadata=metadata
            )

    #
    # def transform_documents(
    #         self, documents: Sequence[Document], **kwargs: Any
    # ) -> Sequence[Document]:
    #     return list(self.lazy_transform_documents(
    #     documents=iter(documents), **kwargs))
    #
    async def alazy_transform_documents(  # type:ignore
        self,
        documents: Union[AsyncIterator[Document], Iterator[Document]],
        **kwargs: Any
    ) -> AsyncIterator[Document]:
        _callbacks = kwargs.get("callbacks", None)
        if isinstance(documents, AsyncIterator):
            async_documents = cast(AsyncIterator[Document], documents)
        else:
            async_documents = to_async_iterator(documents)
        async for doc in async_documents:
            _input = self.get_input(doc)
            output = await self.llm_chain.apredict(callbacks=_callbacks, **_input)
            if not output:
                continue
            metadata = copy.deepcopy(doc.metadata)
            metadata["transformer"] = self.__class__.__name__
            yield Document(
                page_content="SUMMARY:\n" + str(output).strip(), metadata=metadata
            )

    #
    # async def atransform_documents(
    #         self, documents: Sequence[Document], **kwargs: Any
    # ) -> Sequence[Document]:
    #     """Asynchronously transform a list of documents.
    #
    #     Args:
    #         documents: A sequence of Documents to be transformed.
    #
    #     Returns:
    #         A list of transformed Documents.
    #     """
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, partial(self.transform_documents, **kwargs), documents
    #     )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        get_input: Optional[Callable[[Document], dict]] = None,
        llm_chain_kwargs: Optional[dict] = None,
    ) -> "SummarizeTransformer":
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else _default_get_input
        llm_chain = LLMChain(
            llm=llm,
            prompt=_prompt,
            **(llm_chain_kwargs or {}),
        )
        return cls(llm_chain=llm_chain, get_input=_get_input)
