import hashlib
import uuid
from typing import Any, List, Optional, Type, TypeVar, cast, Tuple, Dict, Union, \
    Container

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema import BaseStore
from langchain.schema.document import BaseDocumentTransformer, Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStoreRetriever

from .wrapper_vectorstore import WrapperVectorStore

# %%
VST = TypeVar("VST", bound="VectorStore")


class ParentVectorStore(BaseModel, WrapperVectorStore):
    # Deprecated
    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    docstore: BaseStore[str, Union[Document, List[str]]]
    """The storage layer for the parent documents"""

    doc_id_key: str = "source"
    """The metadata to identify the id of the parents """

    chunk_id_key: str = "_chunk_id"
    """The metadata to identify the chunck. Add an id if the chunk can not have one """

    child_ids_key: str = "_child_ids"
    """Contain a list with the vectorstore id for all 
    corresponding transformed chunk."""

    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    chunk_transformer: Optional[BaseDocumentTransformer]
    """The transformer to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def _get_trunk_from_sub_docs(self, sub_docs: List[Document]) -> List[Document]:
        ids = []
        for d in sub_docs:
            if d.metadata[self.chunk_id_key] not in ids:
                ids.append(d.metadata[self.chunk_id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        if not self.chunk_transformer:
            return self.vectorstore.as_retriever(**kwargs)

        class ParentVectorRetriever(VectorStoreRetriever):
            """Retrieve from a set of multiple embeddings for the same document."""

            def _get_relevant_documents(
                    me,
                    query: str, *,
                    run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                """Get documents relevant to a query.
                Args:
                    query: String to find relevant documents for
                    run_manager: The callbacks handler to use
                Returns:
                    List of relevant documents
                """
                vectorstore = cast(ParentVectorStore, self.vectorstore)
                sub_docs = vectorstore.similarity_search(query, **self.search_kwargs)
                return self._get_trunk_from_sub_docs(sub_docs)

        return ParentVectorRetriever(
            vectorstore=self,
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )

    def add_documents(self, documents: List[Document], *,  # FIXME: lazy ?
                      ids: Optional[List[str]] = None,
                      add_to_docstore: bool = True, **kwargs: Any) -> List[str]:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
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
                    map_doc_ids[doc.metadata[self.doc_id_key]] = id

            else:
                for doc in documents:
                    if self.doc_id_key not in doc.metadata:
                        raise ValueError(
                            "Each document must have a uniq id."
                        )
                    ids = []
                    for doc in documents:
                        # Some docstore refuse some characters in the id.
                        # We convert the id to hash
                        doc_id = doc.metadata[self.doc_id_key]
                        hash_id = hashlib.sha256(
                            str(doc_id).encode("utf-8")).hexdigest()
                        ids.append(hash_id)
                        map_doc_ids[doc_id] = hash_id

        else:
            chunk_ids = ids
            ids = None

        if self.parent_transformer:
            # TODO Check if all documents has en id

            chunk_documents = self.parent_transformer.transform_documents(documents)
        else:
            chunk_documents = documents

        if chunk_ids is None:  # FIXME: vérifier tous les scénarios
            # Generate an id for each chunk, or use the ids
            # Put the associated chunk id the the transformation.
            # Then, it's possible to retrieve the original chunk with this
            # transformation.
            # for chunk in chunk_documents
            #     if self.chunk_id_key not in chunk.metadata:
            #         chunk.metadata[self.chunk_id_key]=str(uuid.uuid4())
            chunk_ids = [
                chunk.metadata.get(self.chunk_id_key, str(uuid.uuid4())) for chunk in
                chunk_documents]
            if not add_to_docstore:  # FIXME
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )  # TODO: verifier si pas 2 fois le même id ?

        chunk_ids_for_doc: Dict[str, List[str]] = {}
        if self.parent_transformer:
            # Associate each chunk with the parent
            for chunk_id, chunk_document in zip(chunk_ids, chunk_documents):
                doc_id = map_doc_ids[chunk_document.metadata[self.doc_id_key]]
                list_of_chunk_ids = chunk_ids_for_doc.get(doc_id, [])
                list_of_chunk_ids.append(chunk_id)
                chunk_ids_for_doc[doc_id] = list_of_chunk_ids

        full_chunk_docs = []
        # TOTRY: on call of chunk_transformer
        if not self.chunk_transformer:
            self.vectorstore.add_documents(documents=chunk_documents, ids=chunk_ids)
        else:
            for chunk_id, chunk_doc in zip(chunk_ids,
                                           chunk_documents):
                all_transformed_chunk: Container[
                    Document] = self.chunk_transformer.transform_documents(
                    [chunk_doc])  # FIXME: un seul
                # If in transformed chunk, add the id of the associated chunk
                for transformed_chunk in all_transformed_chunk:
                    transformed_chunk.metadata[self.chunk_id_key] = chunk_id
                # Save the transformed versions
                transformed_persistance_ids = self.vectorstore.add_documents(
                    all_transformed_chunk)
                # Inject id of transformed ids in the chuck document
                chunk_doc.metadata[self.child_ids_key] = ','.join(
                    transformed_persistance_ids)
                # Prepare the mset in docstore
                full_chunk_docs.append((chunk_id, chunk_doc))

            if add_to_docstore:
                # Add the chunks in docstore.
                # In the retriever, it's this intances to return
                # in metadata[child_ids_key], it's possible to find the id of all
                # transformed versions
                self.docstore.mset(full_chunk_docs)
                # TODO: voir si pas de add_to_docstore

        if self.parent_transformer:
            # With the *parent* mode, for each parent document,
            # we must save the id of all chunk.
            # Then, it's possible to remove/update all chunk when the parent document
            # was updated.
            # Save the parent association wih all chunk  FIXME: flag add_to_docstore
            mset_values: List[Tuple[str, List[str]]] = []
            for parent_id, doc in zip(ids, documents):
                mset_values.append((parent_id, chunk_ids_for_doc[parent_id]))
            self.docstore.mset(mset_values)
            return ids
        else:
            return chunk_ids

    async def aadd_documents(
            self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        # TODO:
        raise NotImplementedError("aadd_documents not implemented")

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if self.parent_transformer:
            if not ids:
                raise ValueError(
                    "ids must be set"
                )
            lists_of_chunk_by_doc_ids = cast(List[List[str]], self.docstore.mget(ids))
            chunk_by_doc_ids = [id for l in lists_of_chunk_by_doc_ids for id in l]
        else:
            chunk_by_doc_ids = ids

        transformed_ids = set()
        if self.chunk_transformer:
            chunk_docs = self.docstore.mget(chunk_by_doc_ids)
            self.docstore.mdelete(chunk_by_doc_ids)
            for chunk_doc in chunk_docs:
                if chunk_doc:
                    transformed_ids.update(
                        chunk_doc.metadata[self.child_ids_key].split(','))
        if transformed_ids:
            self.vectorstore.delete(ids=list(transformed_ids))
        elif self.parent_transformer:
            return self.vectorstore.delete(ids=chunk_by_doc_ids)
        elif not self.parent_transformer and self.chunk_transformer:
            return len(transformed_ids) != 0
        else:
            return self.vectorstore.delete(ids=ids)


    async def adelete(
            self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        # TODO
        raise NotImplementedError("adelete not implemented")

    @classmethod
    def from_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> VST:
        raise NotImplementedError("from_texts not implemented")

    # %% FIXME
    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        subdocs = self.vectorstore.search(query=query, search_type=search_type,
                                          **kwargs)
        return self._get_trunk_from_sub_docs(subdocs)

    async def asearch(
            self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asearch(
            query=query, search_type=search_type, **kwargs
        )

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = self.vectorstore.similarity_search(query=query, k=k, **kwargs)
        return self._get_trunk_from_sub_docs(subdocs)

    def similarity_search_with_score(
            self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        subdocs_and_score = self.vectorstore.similarity_search_with_score(*args,
                                                                          **kwargs)
        # FIXME: et faire les async
        return self._get_trunk_from_sub_docs(subdocs_and_score)

    def similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        subdocs = self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs)

    async def asimilarity_search_with_relevance_scores(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        subdocs = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs)  # FIXME: async

    async def asimilarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = await self.vectorstore.asimilarity_search(query=query, k=k, **kwargs)
        return self._get_trunk_from_sub_docs(subdocs)  # FIXME: async

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = self.vectorstore.similarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs)

    async def asimilarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        subdocs = await self.vectorstore.asimilarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return self._get_trunk_from_sub_docs(subdocs)  # FIXME: async

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
        return self._get_trunk_from_sub_docs(subdocs)

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
        return self._get_trunk_from_sub_docs(subdocs)  # FIXME: async

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
        return self._get_trunk_from_sub_docs(subdocs)

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
        return self._get_trunk_from_sub_docs(subdocs)  # FIXME: async
