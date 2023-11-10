# ruff: noqa
import logging
import os
import pathlib
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from dotenv import load_dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.document_transformers import LongContextReorder
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.indexes import index
from langchain.llms.openai import OpenAI
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
    MultiQueryRetriever,
    SelfQueryRetriever,
    WikipediaRetriever,
)
from langchain.retrievers.document_compressors import (
    CohereRerank,
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma

from langchain_rag.document_transformers import (
    CopyDocumentTransformer,
    DocumentTransformers,
    GenerateQuestionsTransformer,
    SummarizeTransformer,
)
from langchain_rag.vectorstores import RAGVectorStore

# %% set parameters
load_dotenv(override=True)
logger = logging.getLogger(__name__)

id_key = "id"
nb_documents_to_import = 3
top_k = 4

context_size = 4096  # For the demonstration use a smal context_size.
max_tokens = int(context_size * (30 / 100))  # x% for the response
max_input_tokens = (
    context_size - max_tokens
) // top_k  # Need top_k fragment in the prompt

ROOT_PATH = tempfile.gettempdir() + "/rag"
# Clean up
if os.path.exists(ROOT_PATH):
    shutil.rmtree(ROOT_PATH)
os.makedirs(ROOT_PATH)
VS_PATH = ROOT_PATH + "/vs"

# %% Set debug and trace

from langchain.callbacks import StdOutCallbackHandler
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
# %% pprint
CALLBACKS = []


def pretty_print_docs(
    docs: Union[str, List[Document]], metadatas: Sequence[str], kind: str = "Variations"
) -> None:
    def print_metadata(d: Document) -> str:
        s = ",\n".join(
            [f"{metadata}={repr(d.metadata.get(metadata))}" for metadata in metadatas]
        )
        if s:
            return f"\n\033[92m{s}\033[0m"
        return ""

    def print_doc(d: Document, i: int) -> str:
        r = f"\033[94m{kind} {i + 1}:\n{d.page_content[:80]}"
        if len(d.page_content) > 80:
            r += f"...[:{max(0, len(d.page_content) - 80)}]"
        r += f"\033[0m{print_metadata(d)}"
        return r

    if type(docs) is list:
        print(f"\n{'-' * 40}\n".join([print_doc(d, i) for i, d in enumerate(docs)]))
    else:
        print(f"\033[92m{docs}\033[0m")


# %% Select the llm
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.2,
    max_tokens=max_tokens,
)
# langchain.llm_cache = SQLiteCache(database_path=ROOT_PATH + "/cache_llm")

# %% Select the embedding
embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(),
    LocalFileStore(ROOT_PATH + "/cache_embedding"),
    namespace="cache",
)

# %% Test
# CONNECTION_STRING = PGVector.connection_string_from_db_params(
#     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
#     host=os.environ.get("PGVECTOR_HOST", "localhost"),
#     port=int(os.environ.get("PGVECTOR_PORT", "5432")),
#     database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
#     user=os.environ.get("PGVECTOR_USER", "postgres"),
#     password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
# )
#
# CONNECTION_STRING="postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/"
# COLLECTION_NAME = "state_of_the_union_test"
# docs=[Document(page_content="hello word")]
# pg_vectorstore = PGVector.from_documents(
#     embedding=embeddings,
#     documents=docs,
#     collection_name=COLLECTION_NAME,
#     connection_string=CONNECTION_STRING,
# )
# r=pg_vectorstore.as_retriever().get_relevant_documents("hello word")
#
# pg_vectorstore = PGVector(
#     connection_string=CONNECTION_STRING,
#     embedding_function=embeddings,
#     collection_name=COLLECTION_NAME,
#     logger=logger,
#     # connection_string=CONNECTION_STRING,
# )
# pg_vectorstore.add_documents(docs)

# %% Select the transformer
parent_transformer = TokenTextSplitter(chunk_size=max_input_tokens, chunk_overlap=0)
chunk_transformer = DocumentTransformers(
    transformers=[
        GenerateQuestionsTransformer.from_llm(llm),
        SummarizeTransformer.from_llm(llm),
        CopyDocumentTransformer(),
    ]
)

# %% Prepare the persistance
pathlib.Path(ROOT_PATH).mkdir(exist_ok=True)

chroma_vectorstore = Chroma(
    collection_name="all_variations_of_chunks",
    persist_directory=VS_PATH,
    embedding_function=embeddings,
)

# %% SQL factory
# rag_vectorstore, index_kwargs = RAGVectorStore.from_vs_in_sql(
#     vectorstore=chroma_vectorstore,
#     parent_transformer=parent_transformer,
#     chunk_transformer=chunk_transformer,
#     source_id_key="source",
#     db_url=f"sqlite:///{bROOT_PATH}/record_manager.db",
# )
# %% memory factory
rag_vectorstore, index_kwargs = RAGVectorStore.from_vs_in_sql(
    vectorstore=chroma_vectorstore,
    parent_transformer=parent_transformer,
    chunk_transformer=chunk_transformer,
    source_id_key="source",
    db_url=f"sqlite:///{ROOT_PATH}/record_manager.db",
)
# %% Manual factory
# from langchain.storage import EncoderBackedStore
# import pickle
# from langchain.indexes import SQLRecordManager
#
#
# rag_vectorstore = RAGVectorStore(
#     vectorstore=chroma_vectorstore,
#     docstore=EncoderBackedStore[str, Document](
#         store=LocalFileStore(root_path=ROOT_PATH + "/chunks"),
#         key_encoder=lambda x: x,
#         value_serializer=pickle.dumps,
#         value_deserializer=pickle.loads
#     ),
#     source_id_key="source",  # Uniq id of documents
#     parent_transformer=parent_transformer,
#     chunk_transformer=chunk_transformer,
#     search_kwargs={"k": 10}
# )
# engine = sqlalchemy.engine.create_engine(
# url=f"sqlite:///{ROOT_PATH}/record_manager.db")
# index_kwargs = {
#     "record_manager": SQLRecordManager(
#         namespace="record_manager_cache",
#         # db_url=f"sqlite:///{ROOT_PATH}/record_manager.db"
#         engine= engine,
#     ),
#     "vector_store": rag_vectorstore,
#     "source_id_key": "source"
# }
# index_kwargs["record_manager"].create_schema()
# %% Import documents

documents = WikipediaRetriever(
    top_k_results=nb_documents_to_import,
    wiki_client=None,
).get_relevant_documents("mathematic")

index(docs_source=documents, cleanup="incremental", **index_kwargs)

# %% Refine retrievers
merge_retriever = MergerRetriever(
    retrievers=[
        SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=rag_vectorstore,
            document_contents="Documents on mathematics",
            metadata_field_info=[
                AttributeInfo(
                    name="title",
                    description="The title of the document.",
                    type="string",
                ),
            ],
            verbose=True,
        ),
        chroma_vectorstore.as_retriever(
            search_kwargs={"filter": {"transformer": {"$eq": "SummarizeTransformer"}}}
        ),
    ]
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=merge_retriever,
)
# %%  Compress retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=DocumentCompressorPipeline(
        transformers=[
            EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7),
            LongContextReorder(),
        ]
    ),
    base_retriever=multi_query_retriever,
)
final_retriever = compression_retriever
# %% Use the RAG
query = "What is the difference between pure and applied mathematics?"
# %% Use the RetrievalQAWithSourcesChain
# from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
#
# # chain = RetrievalQAWithReferencesAndVerbatimsChain.from_chain_type(
# chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     chain_type="map_reduce",
#     retriever=compression_retriever,
# )
# result = chain(query, callbacks=CALLBACKS)
# print(result['answer'])
# print(result['sources'])
# %% Use RetrievalQAWithReferencesChain
from langchain_qa_with_references.chains import RetrievalQAWithReferencesChain

chain = RetrievalQAWithReferencesChain.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=final_retriever,
    callbacks=CALLBACKS,
)
result = chain(query)
print(result["answer"])
pretty_print_docs(
    cast(List[Document], result["source_documents"]), ["source"], kind="Chunk"
)

# %% Use RetrievalQAWithReferencesAndVerbatimsChain
from langchain_qa_with_references.chains import (
    RetrievalQAWithReferencesAndVerbatimsChain,
)

chain = RetrievalQAWithReferencesAndVerbatimsChain.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=final_retriever,
    callbacks=CALLBACKS,
)
result = chain(query)
print(result["answer"])
pretty_print_docs(
    cast(List[Document], result["source_documents"]), ["source", "verbatims"]
)
print(chain(query)["answer"])
print(chain(query)["answer"])
print(chain(query)["answer"])
print(chain(query)["answer"])
