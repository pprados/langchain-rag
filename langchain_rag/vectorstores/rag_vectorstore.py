import asyncio
import concurrent
import hashlib
import logging
import uuid
from copy import copy, deepcopy
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

from langchain.storage import EncoderBackedStore
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from sqlalchemy import (
    Engine,
    create_engine,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)

from .wrapper_vectorstore import WrapperVectorStore

logger = logging.getLogger(__name__)

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


class RAGVectorStore(WrapperVectorStore):
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

    def __init__(
        self,
        *,
        vectorstore: VectorStore,
        docstore: BaseStore[str, Union[Document, List[str]]],
        source_id_key: Union[str, Callable[[Document], str], None] = "source",
        chunk_id_key: str = "_chunk_id",
        child_ids_key: str = "_child_ids",
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
        chunk_transformer: Optional[BaseDocumentTransformer] = None,
        parent_transformer: Optional[BaseDocumentTransformer] = None,
        coef_trunk_k: int = 3,
    ):
        """

        Parameters
        ----------
        vectorstore         The real vectorstore for saving chunks
        docstore            The storage layer for the parent documents
        source_id_key       The metadata to identify the id of the parents
        chunk_id_key        The metadata to identify the chunk. Add an id if the chunk
                            can not have one
        child_ids_key       Contain a list with the vectorstore id for all corresponding
                            transformed chunk.
        search_type         Type of search to perform. Defaults to "similarity".
        search_kwargs       Keyword arguments to pass to the search function.
        chunk_transformer   The transformer to use to create child documents.
        parent_transformer  The transformer to use to create parent documents.
        coef_trunk_k        If search_kwargs["k"] is not set, use k * coef_trunk_k to
                            loads trunks candidates.
        """
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.source_id_key = source_id_key
        self.chunk_id_key = chunk_id_key
        self.child_ids_key = child_ids_key
        self.search_type = search_type
        self.search_kwargs = search_kwargs or {}
        self.chunk_transformer = chunk_transformer
        self.parent_transformer = parent_transformer
        self.coef_trunk_k = coef_trunk_k

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

    async def _aget_trunk_from_sub_docs(
        self, sub_docs: List[Document], **kwargs: Any
    ) -> List[Document]:
        if self.chunk_transformer:
            ids = []
            for d in sub_docs:
                if d.metadata[self.chunk_id_key] not in ids:
                    ids.append(d.metadata[self.chunk_id_key])
            docs = cast(List[Document], await self.docstore.amget(ids))
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

    async def _aupdate_score_of_chunk(
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
        docs = cast(Sequence[Document], await self.docstore.amget(ids))
        map_ids = {doc.metadata[key]: i for doc, i in zip(docs, ids) if doc}
        return sorted(
            [(d, scores[map_ids[d.metadata[key]]]) for d in docs if d is not None],
            key=lambda x: x[1],
            reverse=True,
        )

    def _get_trunk_from_sub_docs_and_score(
        self, sub_docs_and_score: List[Tuple[Document, float]], k: int, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self.chunk_transformer:
            result = self._update_score_of_chunk(sub_docs_and_score)
        else:
            result = sub_docs_and_score
        if self.search_kwargs.get("k", k) < k:
            raise ValueError("The search_kwargs['k'] must be >= 'k'")
        return result[:k]

    async def _aget_trunk_from_sub_docs_and_score(
        self, sub_docs_and_score: List[Tuple[Document, float]], k: int, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self.chunk_transformer:
            result = await self._aupdate_score_of_chunk(sub_docs_and_score)
        else:
            result = sub_docs_and_score
        if self.search_kwargs.get("k", k) < k:
            raise ValueError("The search_kwargs['k'] must be >= 'k'")
        return result[:k]

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: dict = {}, **kwargs: Any
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
        documents: Sequence[Document],
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
        chunk_documents: Sequence[Document]
        if self.parent_transformer:
            if hasattr(self.parent_transformer, "lazy_transform_documents"):
                chunk_documents = list(
                    self.parent_transformer.lazy_transform_documents(iter(documents))
                )
            else:
                chunk_documents = list(
                    self.parent_transformer.transform_documents(documents)
                )
            if not chunk_documents:
                raise ValueError("The parent_transformer must generate documents")
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
            self.vectorstore.add_documents(
                documents=list(chunk_documents), ids=chunk_ids
            )
        else:
            for chunk_id, chunk_doc in zip(chunk_ids, chunk_documents):
                all_transformed_chunk: Sequence[
                    Document
                ] = self.chunk_transformer.transform_documents(
                    [chunk_doc]
                )  # PPR: transform multiple documents or one by one?
                if not all_transformed_chunk:
                    raise ValueError(
                        "The chunk_transformer must generate documents or"
                        " set chunk_transformer=None"
                    )
                # If in transformed chunk, add the id of the associated chunk
                for transformed_chunk in all_transformed_chunk:
                    transformed_chunk.metadata[self.chunk_id_key] = chunk_id
                # Save the transformed versions
                transformed_persistance_ids = self.vectorstore.add_documents(
                    list(all_transformed_chunk)
                )
                transformed_persistance_ids = [
                    str(id) for id in transformed_persistance_ids
                ]
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
            self.docstore.mset(full_chunk_docs)
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

        chunk_documents: Sequence[Document]
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
        if isinstance(chunk_documents, List):
            list_chunk_documents = cast(List[Document], chunk_documents)
        else:
            list_chunk_documents = list(chunk_documents)
        full_chunk_docs = []
        if not self.chunk_transformer:
            await self.vectorstore.aadd_documents(
                documents=list_chunk_documents, ids=chunk_ids
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
                transformed_persistance_ids = [
                    str(id) for id in transformed_persistance_ids
                ]
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
            self.docstore.mdelete(ids)
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
            await self.docstore.amdelete(ids)
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
        return await self._aget_trunk_from_sub_docs(subdocs, **kwargs)

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
        return await self._aget_trunk_from_sub_docs_and_score(subdocs_and_score, k=k)

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
        return (await self._aupdate_score_of_chunk(subdocs_and_score))[:k]

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
        from langchain_core.indexing import InMemoryRecordManager

        record_manager = InMemoryRecordManager(namespace="in-memory")
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
        **kwargs: Any,
    ) -> Tuple["RAGVectorStore", Dict[str, Any]]:
        from langchain_community.indexes._sql_record_manager import SQLRecordManager

        if isinstance(engine, AsyncEngine):
            assert (
                use_async is not False
            ), "engine is AsyncEngine, use_async must be True or None"
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

        from langchain_community.storage import SQLStore

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

            async def init() -> None:
                await record_manager.acreate_schema()
                await sql_docstore.acreate_schema()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(asyncio.run, init()).result()

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

        return (
            rag_vectorstore,
            {
                "record_manager": record_manager,
                "vector_store": vectorstore,
                "source_id_key": kwargs.get("source_id_key", "source"),
            },
        )

    def __deepcopy__(self, memodict: Optional[dict[int, Any]] = {}) -> "RAGVectorStore":
        new_rag_vectorstore = copy(self)
        new_rag_vectorstore.docstore = copy(self.docstore)
        new_rag_vectorstore.vectorstore = copy(self.vectorstore)
        return new_rag_vectorstore

    @property
    def session_maker(self) -> Any:
        if hasattr(self.docstore, "store"):
            return self.docstore.store.session_factory
        if hasattr(self.vectorstore, "session_factory"):
            return self.vectorstore.session_factory
        if hasattr(self.vectorstore, "session_maker"):
            return self.vectorstore.session_maker
        assert False, "Non session_maker detected"

    def __setattr__(self, key: str, val: Any) -> None:
        if key == "session_maker":
            self._set_session_maker(val)
        else:
            super().__setattr__(key, val)

    def _set_session_maker(self, session_maker: Any) -> None:
        # Manage docstore
        store = self.docstore
        if hasattr(self.docstore, "store"):
            store = self.docstore.store
        if hasattr(store, "session_factory"):
            store.session_factory = session_maker
        elif hasattr(store, "session_maker"):
            store.session_maker = session_maker
        else:
            logger.warning(
                "The docstore has no session_factory or session_maker attribute"
            )

        # Manage vectorstore
        if hasattr(self.vectorstore, "session_factory"):
            self.vectorstore.session_factory = session_maker
        elif hasattr(self.vectorstore, "session_maker"):
            self.vectorstore.session_maker = session_maker
        else:
            logger.warning(
                "The vectorstore has no session_factory or session_maker attribute"
            )

    @staticmethod
    def copy_with_session_maker(
        session_maker: Any,
        rag_vectorstore: "RAGVectorStore",
        index_kwargs: Dict[str, Any],
    ) -> Tuple["RAGVectorStore", Dict[str, Any]]:
        """Duplicate the RAGVectorStore and index_kwargs with a new session_maker."""
        assert isinstance(rag_vectorstore, RAGVectorStore)
        new_rag_vectorstore = deepcopy(rag_vectorstore)
        new_rag_vectorstore.session_maker = session_maker  # type: ignore

        new_record_manager = copy(index_kwargs["record_manager"])
        # new_record_manager = index_kwargs["record_manager"]
        new_record_manager.session_factory = session_maker

        return new_rag_vectorstore, {
            "record_manager": new_record_manager,
            "source_id_key": index_kwargs["source_id_key"],
            "vector_store": new_rag_vectorstore,
        }
