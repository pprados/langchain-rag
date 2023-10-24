import hashlib
import uuid
from typing import Any, List, Optional, Type, TypeVar, cast, Tuple, Dict, Union

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

    chunk_transformer: BaseDocumentTransformer
    """The transformer to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_transformer: Optional[BaseDocumentTransformer] = None
    """The transformer to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        class ParentVectorRetriever(VectorStoreRetriever):
            """Retrieve from a set of multiple embeddings for the same document."""

            def _get_relevant_documents(
                    self,
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
                # We do this to maintain the order of the ids that are returned
                ids = []
                for d in sub_docs:
                    if d.metadata[vectorstore.chunk_id_key] not in ids:
                        ids.append(d.metadata[vectorstore.chunk_id_key])
                docs = vectorstore.docstore.mget(ids)
                return [d for d in docs if d is not None]

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

        # parent_ids = kwargs.get("ids")
        # add_to_docstore = kwargs.get("add_to_docstore", False)
        # FIXME: a débugger
        chunk_for_docs: Dict[str, List[str]] = []
        chunk_ids = None
        map_doc_ids:Dict[Any,str]={}
        if self.parent_transformer:
            if ids:  # It's the parent ids
                if len(documents) != len(ids):
                    raise ValueError(
                        "Got uneven list of documents and ids. "
                        "If `ids` is provided, should be same length as `documents`."
                    )

                # Inject the ids in the parents. Then, each chunk has this id
                # for id, doc in zip(ids, documents):
                #     doc.metadata[self.doc_id_key] = id
                for id, doc in zip(ids, documents):
                    map_doc_ids[doc.metadata[self.doc_id_key]]=id

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
                        hash_id = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
                        ids.append(hash_id)
                        map_doc_ids[doc_id]=hash_id

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
        else:
            chunk_ids = ids

        chunk_ids_for_doc: Dict[str, List[str]] = {}
        if self.parent_transformer:
            # Associate each chunk with the parent
            for chunk_id, chunk_document in zip(chunk_ids, chunk_documents):
                doc_id=map_doc_ids[chunk_document.metadata[self.doc_id_key]]
                chunk_ids = chunk_ids_for_doc.get(doc_id, [])
                chunk_ids.append(chunk_id)
                chunk_ids_for_doc[doc_id]=chunk_ids

        full_chunk_docs = []
        for chunk_id, chunk_doc in zip(chunk_ids, chunk_documents):
            all_transformed_chunk: List[
                Document] = self.chunk_transformer.transform_documents([chunk_doc])
            # In in transformed chunk, add the id of the associated chunk
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
                pass  # FIXME
            if self.parent_transformer:
                hash_ids = []
                for key in ids:
                    hash = hashlib.sha256()
                    hash.update(bytes(key, 'utf-8'))
                    hash_ids.append(hash.hexdigest())
                ids = hash_ids

            parent_documents_metadata: Dict[str, Any] = self.docstore.mget(ids)
            parent_documents_metadata[self.child_ids_key].split(',')

            # # Ids is the *parent* id
            # # Search all child ids with the parent_id_key
            # from langchain.retrievers.self_query.base import _get_builtin_translator
            # from langchain.chains.query_constructor.base import \
            #     StructuredQueryOutputParser
            # for id in ids:
            #     filter=f'eq("{self.parent_id_key}","{id}")'
            #     query=""
            #     escape_query = re.sub(r'([\"])', r'\\\1', query)
            #     escape_filter = re.sub(r'([\"])', r'\\\1', filter)
            #
            #     structured_query = StructuredQueryOutputParser.from_components().parse(
            #         ('```json\n'
            #          '{\n'
            #          # f'"limit":{top_k},\n'
            #          f'"query":"{escape_query}",\n'
            #          f'"filter":"{escape_filter}"\n'
            #          '}\n'
            #          '```'
            #          )
            #     )
            #     translator=_get_builtin_translator(self.vectorstore)
            #     new_query, kwargs_filter = translator.visit_structured_query(structured_query)
            #     self.vectorstore.get(
            #         query="",
            #         search_type="similarity",
            #         **kwargs_filter)

        docs = self.docstore.mget(ids)
        delete_something = False
        for doc in docs:
            if doc:
                delete_something = True
                self.vectorstore.delete(doc.metadata[self.child_ids_key])

        if ids:
            self.docstore.mdelete(ids)
        return delete_something

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
        return self.vectorstore.search(query=query, search_type=search_type, **kwargs)

    async def asearch(
            self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asearch(
            query=query, search_type=search_type, **kwargs
        )

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=k, **kwargs)

    def similarity_search_with_score(
            self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(*args, **kwargs)

    def similarity_search_with_relevance_scores(
            self,
            query: str,
            k: int = 4,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=k, **kwargs
        )

    async def asimilarity_search_with_relevance_scores(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query, k=k, **kwargs
        )

    async def asimilarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asimilarity_search(query=query, k=k, **kwargs)

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.vectorstore.similarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )

    async def asimilarity_search_by_vector(
            self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asimilarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        return self.vectorstore.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        return await self.vectorstore.amax_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        return self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            **kwargs: Any,
    ) -> List[Document]:
        return await self.vectorstore.amax_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )
