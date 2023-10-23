import asyncio
from functools import partial
from typing import Callable, Sequence, Optional, Dict, Any, Generator, List, cast

from langchain.chains import LLMChain
from langchain.output_parsers import NumberedListOutputParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document, BaseDocumentTransformer
from langchain.schema.language_model import BaseLanguageModel
from langchain.pydantic_v1 import BaseModel
from langchain_parent.document_transformers import RunnableDocumentTransformer
from langchain_parent.document_transformers.runnable_document_transformer import \
    RunnableGeneratorDocumentTransformer

from .generate_questions import GenerateQuestions


def _default_get_input(doc: Document) -> Dict[str, Any]:
    """Return the context chain input."""
    return {
        "context": doc.page_content,
    }

class SummarizeAndQuestions(BaseModel):
    summary:str
    """the document summary."""
    questions:List[str]
    """A list of questions"""

_default_parser = PydanticOutputParser(pydantic_object=SummarizeAndQuestions)

_default_template = """
1. Given a text input, generate {nb_of_questions} questions from it in the same language. 
2. Sumarize a text input in the same language. 
Context:
```
{context}
```
{format_instructions}"""



def _get_default_chain_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        template=_default_template,
        output_parser=_default_parser,
        partial_variables={"format_instructions": _default_parser.get_format_instructions()}
    )


class SummarizeAndQuestionsTransformer(RunnableGeneratorDocumentTransformer):
    """Generate questions and summarize for each Documents."""

    def lazy_transform_documents(
            self,
            documents: Sequence[Document],
            **kwargs: Any,
    ) -> Generator[Document, None, None]:
        """Compress page content of raw documents."""
        _callbacks = kwargs.get("callbacks", None)
        for doc in documents:
            _input = self.get_input(doc)
            output = cast(SummarizeAndQuestions,self.llm_chain.predict_and_parse(
                callbacks=_callbacks,
                **{**self.get_input(doc),
                        **{"nb_of_questions": self.nb_of_questions}}
            ))
            if not output:
                continue
            yield Document(page_content=output.summary, metadata=doc.metadata)
            for q in output.questions:
                yield Document(page_content=q, metadata=doc.metadata)

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: Optional[PromptTemplate] = None,
            get_input: Optional[Callable[[Document], dict]] = None,
            nb_of_questions: int = 3,
            llm_chain_kwargs: Optional[dict] = None,
    ) -> 'SummarizeAndQuestionsTransformer':
        """Initialize from LLM."""
        _prompt = prompt if prompt is not None else _get_default_chain_prompt()
        _get_input = get_input if get_input is not None else _default_get_input
        llm_chain = LLMChain(llm=llm, prompt=_prompt, **(llm_chain_kwargs or {}))
        return cls(llm_chain=llm_chain,
                   get_input=_get_input,
                   nb_of_questions=nb_of_questions)
