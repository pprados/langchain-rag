import asyncio
import hashlib
import threading
import uuid
from asyncio import current_task, Task
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from sqlalchemy import (
    Connection,
    Engine,
    create_engine,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine, async_sessionmaker, async_scoped_session, AsyncSession,
    AsyncConnection,
)
from sqlalchemy.orm import scoped_session, sessionmaker

from .wrapper_vectorstore import WrapperVectorStore

# %%
VST = TypeVar("VST", bound="VectorStore")


def _get_source_id_assigner(
        source_id_key: Union[str, Callable[[Document], str], None],
) -> Callable[[Document], Union[str, None]]:
    """Get the source id from the document."""
    if source_id_key is None:
        return lambda doc: None
    elif isinstance(source_id_key, str):
        return lambda doc: doc.metadata[source_id_key]
    elif callable(source_id_key):
        return source_id_key
    else:
        raise ValueError(
            f"source_id_key should be either None, a string or a callable. "
            f"Got {source_id_key} of type {type(source_id_key)}."
        )


class RAGVectorStore(BaseModel, WrapperVectorStore):
    """Retrieve small chunks then retrieve their parent documents.

    When splitting documents for retrieval, there are often conflicting desires:

    1. You may want to have small documents, so that their embeddings can most
        accurately reflect their meaning. If too long, then the embeddings can
        lose meaning.
    2. You want to have long enough documents that the context of each chunk is
        retained.

    The ParentDocumentRetriever strikes that balance by splitting and storing
    small chunks of data. During retrieval, it first fetches the small chunks
    but then looks up the parent ids for those chunks and returns those larger
    documents.

    Note that "parent document" refers to the document that a small chunk
    originated from. This can either be the whole raw document OR a larger
    chunk.

    Examples:

        .. code-block:: python

            # Imports
            from langchain_community.vectorstores import Chroma
            from langchain_openai.embeddings import OpenAIEmbeddings
            from langchain_core.text_splitter import RecursiveCharacterTextSplitter
            from langchain.storage import InMemoryStore

            # This text splitter is used to create the parent documents
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
            # This text splitter is used to create the child documents
            # It should create documents smaller than the parent
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
            # The storage layer for the parent documents
            store = InMemoryStore()

            # Initialize the retriever
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
    """

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    vectorstore: VectorStore
    """The real vectorstore for saving chunks"""
    docstore: BaseStore[str, Union[Document, List[str]]]
    """The storage layer for the parent documents"""

    source_id_key: Union[str, Callable[[Document], str], None] = "source"
    """The metadata to identify the id of the parents """

    chunk_id_key: str = "_chunk_id"
    """The metadata to identify the chunk. Add an id if the chunk can not have one """

    child_ids_key: str = "_child_ids"
    """Contain a list with the vectorstore id for all 
    corresponding transformed chunk."""

    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    chunk_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create child documents."""

    parent_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    coef_trunk_k: int = 3
    """If search_kwargs["k"] is not set, use k * coef_trunk_k to loads trunks
    candidates.
    """

    def _get_trunk_from_sub_docs(
            self, sub_docs: List[Document], **kwargs: Any
    ) -> List[Document]:
        if self.chunk_transformer:
            ids = []
            for d in sub_docs:
                if d.metadata[self.chunk_id_key] not in ids:
                    ids.append(d.metadata[self.chunk_id_key])
            docs = cast(List[Document], self.docstore.mget(ids))
            result = [d for d in docs if d is not None]
        else:
            result = sub_docs
        if "k" in kwargs:
            return result[: kwargs["k"]]
        else:
            return result

    def _update_score_of_chunk(
            self, sub_chunks_and_score: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        if not self.chunk_transformer:
            return sub_chunks_and_score
        ids = []
        scores: Dict[str, float] = {}
        key = self.chunk_id_key
        for d, s in sub_chunks_and_score:
            if d.metadata[key] not in ids:
                id = d.metadata[key]
                ids.append(id)
                chunk_s = scores.get(id, 1.0)
                scores[id] = min(chunk_s, s)
        docs = cast(Sequence[Document], self.docstore.mget(ids))
        map_ids = {doc.metadata[key]: i for doc, i in zip(docs, ids) if doc}
        return sorted(
            [(d, scores[map_ids[d.metadata[key]]]) for d in docs if d is not None],
            key=lambda x: x[1],
            reverse=True,
        )

    def _get_trunk_from_sub_docs_and_score(
            self, sub_docs_and_score: List[Tuple[Document, float]], k: int,
            **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self.chunk_transformer:
            result = self._update_score_of_chunk(sub_docs_and_score)
        else:
            result = sub_docs_and_score
        if self.search_kwargs.get("k", k) < k:
            raise ValueError("The search_kwargs['k'] must be >= 'k'")
        return result[:k]

    def as_retriever(
            self, search_type: str = "similarity", search_kwargs: dict = {},
            **kwargs: Any
    ) -> VectorStoreRetriever:
        if not self.chunk_transformer:
            return self.vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs, *kwargs
            )

        retriever = VectorStoreRetriever(
            vectorstore=self, search_type=search_type, search_kwargs=search_kwargs
        )
        return retriever

    def add_documents(
            self,
            documents: List[Document],
            *,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
        """
        source_id_key_get = _get_source_id_assigner(self.source_id_key)
        chunk_ids = None
        map_doc_ids: Dict[Any, str] = {}
        if self.parent_transformer:
            if ids:  # It's the parent ids
                if len(documents) != len(ids):
                    raise ValueError(
                        "Got uneven list of documents and ids. "
                        "If `ids` is provided, should be same length as `documents`."
                    )

                for id, doc in zip(ids, documents):
                    key = source_id_key_get(doc)
                    if key in map_doc_ids:
                        raise ValueError(
                            f"Got multiple documents with the same ids `{key}"
                        )
                    map_doc_ids[key] = id

            else:
                ids = []
                for doc in documents:
                    # Some docstore refuse some characters in the id.
                    # We convert the id to hash
                    doc_id = source_id_key_get(doc)
                    hash_id = hashlib.sha256(str(doc_id).encode("utf-8")).hexdigest()
                    ids.append(hash_id)
                    map_doc_ids[doc_id] = hash_id
            self.delete(ids=list(map_doc_ids.values()))
        else:
            chunk_ids = ids
            if chunk_ids:
                self.delete(ids=chunk_ids)
            ids = None

        if self.parent_transformer:
            if hasattr(self.parent_transformer, "lazy_transform_documents"):
                chunk_documents = list(
                    self.parent_transformer.lazy_transform_documents(iter(documents))
                )
            else:
                chunk_documents = list(
                    self.parent_transformer.transform_documents(documents)
                )
        else:
            chunk_documents = documents

        if chunk_ids is None:
            # Generate an id for each chunk, or use the ids
            # Put the associated chunk id after the transformation.
            # Then, it's possible to retrieve the original chunk with this
            # transformation.
            # for chunk in chunk_documents
            #     if self.chunk_id_key not in chunk.metadata:
            #         chunk.metadata[self.chunk_id_key]=str(uuid.uuid4())
            chunk_ids = [
                chunk.metadata.get(self.chunk_id_key, str(uuid.uuid4()))
                for chunk in chunk_documents
            ]

        chunk_ids_for_doc: Dict[str, List[str]] = {}
        if self.parent_transformer:
            # Associate each chunk with the parent
            for chunk_id, chunk_document in zip(chunk_ids, chunk_documents):
                doc_id = map_doc_ids[source_id_key_get(chunk_document)]
                list_of_chunk_ids = chunk_ids_for_doc.get(doc_id, [])
                list_of_chunk_ids.append(chunk_id)
                chunk_ids_for_doc[doc_id] = list_of_chunk_ids
                if self.chunk_id_key not in chunk_document.metadata:
                    chunk_document.metadata[self.chunk_id_key] = chunk_id

        full_chunk_docs = []
        if not self.chunk_transformer:
            self.vectorstore.add_documents(documents=chunk_documents, ids=chunk_ids)
        else:
            for chunk_id, chunk_doc in zip(chunk_ids, chunk_documents):
                all_transformed_chunk: Sequence[
                    Document
                ] = self.chunk_transformer.transform_documents(
                    [chunk_doc]
                )  # PPR: use multiple documents?
                # If in transformed chunk, add the id of the associated chunk
                for transformed_chunk in all_transformed_chunk:
                    transformed_chunk.metadata[self.chunk_id_key] = chunk_id
                # Save the transformed versions
                transformed_persistance_ids = self.vectorstore.add_documents(
                    list(all_transformed_chunk)
                )
                # Inject id of transformed ids in the chuck document
                chunk_doc.metadata[self.child_ids_key] = ",".join(
                    transformed_persistance_ids
                )
                # Prepare the mset in docstore
                full_chunk_docs.append((chunk_id, chunk_doc))

            # Add the chunks in docstore.
            # In the retriever, it's this instances to return
            # in metadata[child_ids_key], it's possible to find the id of all
            # transformed versions
            self.docstore.mset(full_chunk_docs)

        if self.parent_transformer:
            # With the *parent* mode, for each parent document,
            # we must save the id of all chunk.
            # Then, it's possible to remove/update all chunk when the parent document
            # was updated.
            # Save the parent association with all chunk
            ids = cast(List[str], ids)
            mset_values: List[Tuple[str, List[str]]] = []
            for parent_id, doc in zip(ids, documents):
                mset_values.append((parent_id, chunk_ids_for_doc[parent_id]))
            self.docstore.mset(mset_values)
            return ids
        else:
            return chunk_ids

    async def aadd_documents(
            self,
            documents: List[Document],
            *,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
        """
        source_id_key_get = _get_source_id_assigner(self.source_id_key)
        chunk_ids = None
        map_doc_ids: Dict[Any, str] = {}
        if self.parent_transformer:
            if ids:  # It's the parent ids
                if len(documents) != len(ids):
                    raise ValueError(
                        "Got uneven list of documents and ids. "
                        "If `ids` is provided, should be same length as `documents`."
                    )

                for id, doc in zip(ids, documents):
                    map_doc_ids[source_id_key_get(doc)] = id

            else:
                for doc in documents:
                    ids = []
                    for doc in documents:
                        # Some docstore refuse some characters in the id.
                        # We convert the id to hash
                        doc_id = source_id_key_get(doc)
                        hash_id = hashlib.sha256(
                            str(doc_id).encode("utf-8")
                        ).hexdigest()
                        ids.append(hash_id)
                        map_doc_ids[doc_id] = hash_id
            await self.adelete(ids=list(map_doc_ids.values()))
        else:
            chunk_ids = ids
            if chunk_ids:
                await self.adelete(ids=chunk_ids)
            ids = None

        if self.parent_transformer:
            if hasattr(self.parent_transformer, "alazy_transform_documents"):
                chunk_documents = [
                    doc
                    async for doc in self.parent_transformer.alazy_transform_documents(
                        documents
                    )
                ]
            else:
                chunk_documents = await self.parent_transformer.atransform_documents(
                    documents
                )
        else:
            chunk_documents = documents

        if chunk_ids is None:
            # Generate an id for each chunk, or use the ids
            # Put the associated chunk id after the transformation.
            # Then, it's possible to retrieve the original chunk with this
            # transformation.
            chunk_ids = [
                chunk.metadata.get(self.chunk_id_key, str(uuid.uuid4()))
                for chunk in chunk_documents
            ]

        chunk_ids_for_doc: Dict[str, List[str]] = {}
        if self.parent_transformer:
            # Associate each chunk with the parent
            for chunk_id, chunk_document in zip(chunk_ids, chunk_documents):
                doc_id = map_doc_ids[source_id_key_get(chunk_document)]
                list_of_chunk_ids = chunk_ids_for_doc.get(doc_id, [])
                list_of_chunk_ids.append(chunk_id)
                chunk_ids_for_doc[doc_id] = list_of_chunk_ids
                if self.chunk_id_key not in chunk_document.metadata:
                    chunk_document.metadata[self.chunk_id_key] = chunk_id

        full_chunk_docs = []
        if not self.chunk_transformer:
            await self.vectorstore.aadd_documents(
                documents=chunk_documents, ids=chunk_ids
            )
        else:
            for chunk_id, chunk_doc in zip(chunk_ids, chunk_documents):
                all_transformed_chunk: Sequence[
                    Document
                ] = await self.chunk_transformer.atransform_documents(
                    [chunk_doc]
                )  # PPR: use multiple documents?
                # If in transformed chunk, add the id of the associated chunk
                for transformed_chunk in all_transformed_chunk:
                    transformed_chunk.metadata[self.chunk_id_key] = chunk_id
                # Save the transformed versions
                transformed_persistance_ids = await self.vectorstore.aadd_documents(
                    list(all_transformed_chunk)
                )
                # Inject id of transformed ids in the chuck document
                chunk_doc.metadata[self.child_ids_key] = ",".join(
                    transformed_persistance_ids
                )
                # Prepare the mset in docstore
                full_chunk_docs.append((chunk_id, chunk_doc))

            # Add the chunks in docstore.
            # In the retriever, it's this instances to return
            # in metadata[child_ids_key], it's possible to find the id of all
            # transformed versions
            await self.docstore.amset(full_chunk_docs)

        if self.parent_transformer:
            # With the *parent* mode, for each parent document,
            # we must save the id of all chunk.
            # Then, it's possible to remove/update all chunk when the parent document
            # was updated.
            # Save the parent association with all chunk
            ids = cast(List[str], ids)
            mset_values: List[Tuple[str, List[str]]] = []
            for parent_id, doc in zip(ids, documents):
                mset_values.append((parent_id, chunk_ids_for_doc[parent_id]))
            await self.docstore.amset(mset_values)
            return ids
        else:
            return chunk_ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            raise ValueError("ids must be set")
        if self.parent_transformer:
            if not ids:
                raise ValueError("ids must be set")
            lists_of_chunk_by_doc_ids = cast(List[List[str]], self.docstore.mget(ids))
            chunk_by_doc_ids: List[str] = []
            for list_of_ids in lists_of_chunk_by_doc_ids:
                if list_of_ids:
                    chunk_by_doc_ids.extend([id for id in list_of_ids])
        else:
            chunk_by_doc_ids = ids

        if not any(chunk_by_doc_ids):
            return False

        transformed_ids = set()
        if self.chunk_transformer:
            chunk_docs = cast(List[Document], self.docstore.mget(chunk_by_doc_ids))
            self.docstore.mdelete(chunk_by_doc_ids)
            for chunk_doc in chunk_docs:
                if chunk_doc:
                    transformed_ids.update(
                        chunk_doc.metadata[self.child_ids_key].split(",")
                    )
        if transformed_ids:
            self.vectorstore.delete(ids=list(transformed_ids))
        elif self.parent_transformer:
            return self.vectorstore.delete(ids=chunk_by_doc_ids)
        elif not self.parent_transformer and self.chunk_transformer:
            return len(transformed_ids) != 0
        else:
            return self.vectorstore.delete(ids=ids)
        return False

    async def adelete(
            self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        if not ids:
            raise ValueError("ids must be set")
        if self.parent_transformer:
            if not ids:
                raise ValueError("ids must be set")
            lists_of_chunk_by_doc_ids = cast(
                List[List[str]], await self.docstore.amget(ids)
            )
            chunk_by_doc_ids: List[str] = []
            for list_of_ids in lists_of_chunk_by_doc_ids:
                if list_of_ids:
                    chunk_by_doc_ids.extend([id for id in list_of_ids])
        else:
            chunk_by_doc_ids = ids

        if not any(chunk_by_doc_ids):
            return False

        transformed_ids = set()
        if self.chunk_transformer:
            chunk_docs = cast(
                List[Document], await self.docstore.amget(chunk_by_doc_ids)
            )
            await self.docstore.amdelete(chunk_by_doc_ids)
            for chunk_doc in chunk_docs:
                if chunk_doc:
                    transformed_ids.update(
                        chunk_doc.metadata[self.child_ids_key].split(",")
                    )
        if transformed_ids:
            await self.vectorstore.adelete(ids=list(transformed_ids))
        elif self.parent_transformer:
            return await self.vectorstore.adelete(ids=chunk_by_doc_ids)
        elif not self.parent_transformer and self.chunk_transformer:
            return len(transformed_ids) != 0
        else:
            return await self.vectorstore.adelete(ids=ids)
        return False

    @classmethod
    def from_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> VST:
        raise NotImplementedError("from_texts not implemented")

    # %% searches
    def _trunk_k(
            self, result: List[Document], kwargs: Dict[str, Any]
    ) -> List[Document]:
        if "k" in kwargs:
            return result[: kwargs["k"]]
        else:
            return result

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs = self.vectorstore.search(
            query=query, search_type=search_type, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, **kwargs)

    async def asearch(
            self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs = await self.vectorstore.asearch(
            query=query, search_type=search_type, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, **kwargs)

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.search(query=query, search_type="similarity", k=k, **kwargs)

    async def asimilarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await self.asearch(query=query, search_type="similarity", k=k, **kwargs)

    def similarity_search_with_score(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        _search_kwargs = {**kwargs, **self.search_kwargs}

        trunk_k: Optional[int] = self.search_kwargs.get("k")
        if not trunk_k:
            _search_kwargs["k"] = k * self.coef_trunk_k

        subdocs_and_score = self.vectorstore.similarity_search_with_score(
            query=query, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs_and_score(subdocs_and_score, k=k)

    async def asimilarity_search_with_score(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        _search_kwargs = {**kwargs, **self.search_kwargs}

        trunk_k: Optional[int] = self.search_kwargs.get("k")
        if not trunk_k:
            _search_kwargs["k"] = k * self.coef_trunk_k

        subdocs_and_score = await self.vectorstore.asimilarity_search_with_score(
            query=query, **_search_kwargs
        )
        return self._get_trunk_from_sub_docs_and_score(subdocs_and_score, k=k)

    def similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs_and_score = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, **_search_kwargs
        )
        return self._update_score_of_chunk(subdocs_and_score)[:k]

    async def asimilarity_search_with_relevance_scores(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        _search_kwargs = {**kwargs, **self.search_kwargs}
        subdocs_and_score = (
            await self.vectorstore.asimilarity_search_with_relevance_scores(
                query=query, **_search_kwargs
            )
        )
        return self._update_score_of_chunk(subdocs_and_score)[:k]

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = self.vectorstore.similarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    async def asimilarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = await self.vectorstore.asimilarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        subdocs = self.vectorstore.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    async def amax_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        subdocs = await self.vectorstore.amax_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(
            subdocs,
            k=k,
        )

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        subdocs = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs, k=k)

    async def amax_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        subdocs = await self.vectorstore.amax_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs)

    @staticmethod
    def from_vs_in_memory(
            vectorstore: VectorStore,
            *,
            chunk_transformer: Optional[BaseDocumentTransformer] = None,
            parent_transformer: Optional[BaseDocumentTransformer] = None,
            source_id_key: Union[str, Callable[[Document], str]] = "source",
            **kwargs: Any,
    ) -> Tuple["RAGVectorStore", Dict[str, Any]]:
        from langchain.storage import InMemoryStore

        from ..indexes.memory_recordmanager import MemoryRecordManager

        record_manager = MemoryRecordManager(namespace="in-memory")
        docstore = InMemoryStore()
        vectorstore = RAGVectorStore(
            vectorstore=vectorstore,
            docstore=docstore,
            parent_transformer=parent_transformer,
            chunk_transformer=chunk_transformer,
            **kwargs,
        )
        return (
            vectorstore,
            {
                "record_manager": record_manager,
                "vector_store": vectorstore,
                "source_id_key": source_id_key,
            },
        )

    def session_maker(self, bind: Engine | Connection):
        return scoped_session(sessionmaker(bind=bind))

    @staticmethod
    def update_thread_session_factory(bind: Engine | Connection) -> session_maker:
        context = RAGVectorStore._thread_session_factory
        if not hasattr(context, "sessionfactory"):
            context.sessionfactory = scoped_session(sessionmaker(bind=bind))
        return context.sessionfactory

    @staticmethod
    def update_async_session_factory(bind: AsyncEngine | AsyncConnection) -> async_scoped_session:

        context = RAGVectorStore._async_session_factory
        old_sessionfactory = context.get()
        if not old_sessionfactory:
            context.set(async_scoped_session(async_sessionmaker(bind=bind),
                                                          scopefunc=current_task))
        return cast(async_scoped_session,context.get())

    _thread_session_factory = threading.local()
    _async_session_factory = ContextVar("async_sessionfactory", default=None)

    @staticmethod
    @contextmanager
    def index_session_factory(engine: Engine):
        """
        Create an outer session factory, to index() a list of documents, with only
        one SQL transaction.
        If you use this context manager, with data
        Sample:
        ```
        echo = True
        db_url = "postgresql+psycopg://postgres:password_postgres@localhost:5432/"
        engine = create_engine(db_url,echo=echo)
        embeddings = FakeEmbeddings()
        pgvector = PGVector(
            embeddings=embeddings,
            connection=engine,
            engine_args={"echo": echo},
        )

        rag_vectorstore, index_kwargs = RAGVectorStore.from_vs_in_sql(
            vectorstore=pgvector,
            engine=engine,
        )
        # Import all the data in one transaction. All the database will be stables.
        with RAGVectorStore.index_session_factory(engine) as session:
            loader = CSVLoader(
                    "data/faq/faq.csv",
                    source_column="source",
                    autodetect_encoding=True,
                )
            result = index(
                docs_source=loader,
                cleanup="incremental",
                **index_kwargs,
            )
            session.commit()  # Commit all the import or rollback all
        ```

        Args:
            engine: The SQL engine to use.

        Yield:
            session
        """
        # Use an async_scoped_session associated with the connection.
        # Then, it's possible to merge the inner transaction with the outer transaction.
        connection = engine.connect()
        session_factory = RAGVectorStore.update_thread_session_factory(connection)
        outer_transaction = connection.begin()
        yield session_factory()
        RAGVectorStore.update_thread_session_factory(engine)
        outer_transaction.commit()
        outer_transaction.close()
        connection.close()

    @staticmethod
    @asynccontextmanager
    async def aindex_session_factory(engine: AsyncEngine):
        # Use an async_scoped_session associated with the connection.
        # Then, it's possible to merge the inner transaction with the outer transaction.
        connection = await engine.connect()
        session_factory = RAGVectorStore.update_async_session_factory(connection)
        outer_transaction = await connection.begin()
        yield session_factory()
        RAGVectorStore.update_async_session_factory(engine)
        await outer_transaction.commit()  # FIXME: est-ce un doublon ?
        await outer_transaction.close()
        await connection.close()

    @staticmethod
    def from_vs_in_sql(
            vectorstore: VectorStore,
            *,
            engine: Union[None, Engine, AsyncEngine] = None,
            engine_kwargs: Optional[Dict[str, Any]] = None,
            db_url: Optional[str] = None,
            use_async: Optional[bool] = None,
            namespace: str = "rag_vectorstore",
            chunk_transformer: Optional[BaseDocumentTransformer] = None,
            parent_transformer: Optional[BaseDocumentTransformer] = None,
            session_factory: Callable | None = None,
            **kwargs: Any,
    ) -> Tuple["RAGVectorStore", Dict[str, Any]]:
        from langchain.indexes import SQLRecordManager

        from patch_langchain.storage import EncoderBackedStore
        if isinstance(engine, AsyncEngine):
            assert use_async is not False, "engine is AsyncEngine, use_async must be True or None"
            use_async = True  # Force async mode
        if use_async is None:
            use_async = False

        docstore: BaseStore[str, Union[Document, List[str]]]
        if not db_url and not engine:
            raise ValueError("Set db_url or engine")

        if db_url:
            if use_async:
                engine = create_async_engine(url=str(db_url), **(engine_kwargs or {}))
            else:
                engine = create_engine(url=str(db_url), **(engine_kwargs or {}))

        import pickle

        from ..storage.sql_docstore import SQLStore


        record_manager = SQLRecordManager(
            namespace=namespace,
            engine=engine,
            engine_kwargs=engine_kwargs,
            async_mode=use_async,
        )
        sql_docstore = SQLStore(
            namespace=namespace,
            engine=engine,
            engine_kwargs=engine_kwargs,
            async_mode=use_async,
        )
        if not use_async:
            record_manager.create_schema()
            sql_docstore.create_schema()
        else:

            async def init():
                print(f"debug init {threading.current_thread()}")
                await record_manager.acreate_schema()
                await sql_docstore.acreate_schema()
                print(f"fin init {threading.current_thread()}")

            try:
                loop = asyncio.get_event_loop()
                print(f"Trouve une loop {loop} {threading.current_thread()}")  # FIXME
            except RuntimeError:
                loop = asyncio.new_event_loop()
                print(f"Create une loop {loop}")  # FIXME
                asyncio.set_event_loop(loop)
            loop.run_until_complete(init())
        docstore = EncoderBackedStore[str, Union[Document, List[str]]](
            store=sql_docstore,
            key_encoder=lambda x: x,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads,
        )
        rag_vectorstore = RAGVectorStore(
            vectorstore=vectorstore,
            docstore=docstore,
            parent_transformer=parent_transformer,
            chunk_transformer=chunk_transformer,
            **kwargs,
        )

        # Align all the sessions factories
        # FIXME: Ajouter des TU pour les 2 scénarios
        if use_async:
            class async_local_sessionmanager(async_sessionmaker):
                # Need a subclass of async_sessionmaker
                def __init__(self):
                    super().__init__(bind=None)

                def __call__(self, **local_kw: Any) -> AsyncSession:
                    # The default implementation, use an async_scoped_session
                    # associated with the engine (and not with the connexion)
                    session_factory = RAGVectorStore.update_async_session_factory(engine)

                    return session_factory()

            if not session_factory:
                session_factory = async_local_sessionmanager()

            record_manager.session_factory = session_factory
            sql_docstore.session_factory = session_factory
            if hasattr(vectorstore, "session_maker"):
                vectorstore.session_maker = session_factory
            if hasattr(vectorstore, "session_factory"):
                vectorstore.session_factory = session_factory
        else:
            def local_session():
                # The default implementation, use an async_scoped_session
                # associated with the engine (and not with the connexion)
                context = RAGVectorStore._thread_session_factory
                if not hasattr(context, "sessionfactory"):
                    context.sessionfactory = scoped_session(sessionmaker(bind=engine))
                session = context.sessionfactory()
                return session

            if not session_factory:
                # session_factory=scoped_session(sessionmaker(bind=engine))
                session_factory = local_session

            record_manager.session_factory = session_factory
            sql_docstore.session_factory = session_factory
            if hasattr(vectorstore, "session_maker"):
                vectorstore.session_maker = session_factory
            if hasattr(vectorstore, "session_factory"):
                vectorstore.session_factory = session_factory

        return (
            rag_vectorstore,
            {
                "record_manager": record_manager,
                "vector_store": vectorstore,
                "source_id_key": kwargs.get("source_id_key", "source"),
            },
        )
