import os
import shutil
import tempfile

import langchain
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.indexes import index
from langchain.llms.openai import OpenAI
from langchain.retrievers import WikipediaRetriever, MergerRetriever, \
    SelfQueryRetriever, MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, \
    EmbeddingsFilter, CohereRerank
from langchain.storage import LocalFileStore
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma

from langchain_parent.document_transformers import DocumentTransformers, \
    SummarizeTransformer, CopyDocumentTransformer, GenerateQuestionsTransformer
from langchain_parent.vectorstores import RAGVectorStore

# %% set parameters
load_dotenv(override=True)

id_key = 'id'
top_k = 4
nb_documents_to_import = 1

context_size = 512  # For the demonstration use a smal context_size.
max_tokens = int(context_size * (10 / 100))  # 10% for the response
max_input_tokens = context_size - max_tokens

ROOT_PATH = tempfile._gettempdir() + "/rag"
# Clean up
if os.path.exists(ROOT_PATH):
    shutil.rmtree(ROOT_PATH)
os.makedirs(ROOT_PATH)
VS_PATH = ROOT_PATH + "/vs"

# %% Select the llm
llm = OpenAI(temperature=0.5)
langchain.llm_cache = SQLiteCache(database_path=ROOT_PATH + "/cache_llm")

# %% Select the embedding
embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(),
    LocalFileStore(ROOT_PATH + "/cache_embedding"),
    namespace="cache"
)

# %% Select the transformer
parent_transformer = TokenTextSplitter(
    chunk_size=max_input_tokens,
    chunk_overlap=0
)
chunk_transformer = DocumentTransformers(
    transformers=[
        GenerateQuestionsTransformer.from_llm(llm),
        SummarizeTransformer.from_llm(llm),
        CopyDocumentTransformer(),
    ]
)

# %% Prepare the persistance
if not os.path.exists(ROOT_PATH):
    os.makedirs(ROOT_PATH)

chroma_vectorstore = Chroma(
    collection_name="all_variations_of_chunks",
    persist_directory=VS_PATH,
    embedding_function=embeddings,
)

vectorstore, index_kwargs = RAGVectorStore.from_vs_in_sql(
    vectorstore=chroma_vectorstore,
    parent_transformer=parent_transformer,
    chunk_transformer=chunk_transformer,
    source_id_key="source",
    db_url=f"sqlite:///{ROOT_PATH}/record_manager.db",
)
# # %% Import documents
documents = (WikipediaRetriever(top_k_results=nb_documents_to_import)
             .get_relevant_documents("mathematic"))

index(
    docs_source=documents,
    cleanup="incremental",
    **index_kwargs
)

# %% Refine retrievers

retriever = MergerRetriever(
    retrievers=
    [
        SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents="Documents on mathematics",
            metadata_field_info=[
                AttributeInfo(
                    name="title",
                    description="The title of the document.",
                    type="string",
                ),
            ],
            verbose=True),
        chroma_vectorstore.as_retriever(
            search_kwargs={
                "filter": {"transformer": {"$eq": "SummarizeTransformer"}}})
    ])

retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=retriever,
)
# %%  Compress retriever
retriever = ContextualCompressionRetriever(
    base_compressor=DocumentCompressorPipeline(
        transformers=[
            EmbeddingsFilter(
                embeddings=embeddings,
                similarity_threshold=0.7
            ),
            CohereRerank(top_n=top_k),
        ]
    ),
    base_retriever=retriever
)

# %% Set debug and trace
from langchain.callbacks import StdOutCallbackHandler
from pprint import pprint
from typing import *

from langchain.globals import set_debug, set_verbose

set_debug(False)
set_verbose(False)
if True:
    VERBOSE_INPUT = True
    VERBOSE_OUTPUT = True


    class ExStdOutCallbackHandler(StdOutCallbackHandler):
        def on_text(
                self,
                text: str,
                color: Optional[str] = None,
                end: str = "",
                **kwargs: Any,
        ) -> None:
            if VERBOSE_INPUT:
                print("====")
                super().on_text(text=text, color=color, end=end)

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            """Ajoute une trace des outputs du llm"""
            if VERBOSE_OUTPUT:
                print("\n\033[1m> Finished chain with\033[0m")
                knows_keys = {
                    "answer",
                    "output_text",
                    "text",
                    "result",
                    "outputs",
                    "output",
                }
                if "outputs" in outputs:
                    print("\n\033[33m")
                    print(
                        "\n---\n".join(
                            [text["text"].strip() for text in outputs["outputs"]]
                        )
                    )
                    print("\n\033[0m")
                elif knows_keys.intersection(outputs):
                    # Prend la première cles en intersection
                    print(
                        f"\n\033[33m{outputs[next(iter(knows_keys.intersection(outputs)))]}\n\033[0m"
                    )
                else:
                    pass


    CALLBACKS = [ExStdOutCallbackHandler()]
else:
    CALLBACKS = []
# %% Use the RAG
query = "What is the difference between pure and applied mathematics?"
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# chain = RetrievalQAWithReferencesAndVerbatimsChain.from_chain_type(
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever(),
)
result = chain(query, callbacks=CALLBACKS)
print(result['answer'])
pprint([doc.metadata['source'] for doc in result['source_documents']])
