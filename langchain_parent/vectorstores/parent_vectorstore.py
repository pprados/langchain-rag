import hashlib
import uuid
from typing import Any, List, Optional, Type, TypeVar, cast, Tuple, Dict

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

    # record_manager:RecordManager
    docstore: BaseStore[str, Document]
    """The storage layer for the parent documents"""
    parent_id_key: str = "source"
    """The metadata to identify the id of the parents """
    chunk_id_key: str = "_chuck_id"
    """The metadata to identify the bucket """
    all_transformed_id_key: str = "_var_id"
    """Contain a list with the vectorstore id for all 
    corresponding transformed chunk."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    child_transformer: BaseDocumentTransformer
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
        if self.parent_transformer:
            if ids:
                # The ids is for the parents
                if len(documents) != len(ids):
                    raise ValueError(
                        "Got uneven list of documents and ids. "
                        "If `ids` is provided, should be same length as `documents`."
                    )
                # Inject the ids in the parents
                for id, doc in zip(ids, documents):
                    doc.metadata[self.parent_id_key] = id
                ids = None  # Now, generate ids for chunks
            # all_transformed_id_key
            # all_chunk:List[Document]=[]
            # for doc in documents:
            #     _chunks = self.parent_transformer.transform_documents([documents])
            #     for _chunk in _chunks:
            #         _chunk.metadata[self.parent_id_key]=

        if self.parent_transformer:
            # TODO Check if all documents has en id

           chunk_documents = self.parent_transformer.transform_documents(documents)
        else:
            chunk_documents = documents

        if ids is None:  # FIXME: vérifier tous les scénarios
            # Generate an id for each chunk, or use the ids
            chunk_ids = [
                doc.metadata[
                    self.chunk_id_key] if self.chunk_id_key in doc.metadata else str(
                    uuid.uuid4()) for doc in
                chunk_documents]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )

        full_docs = []
        for i, doc in enumerate(chunk_documents):
            _chunk_id = chunk_ids[i]  # FIXME: //
            transformed_docs:List[Document] = self.child_transformer.transform_documents([doc])
            for _transformed_doc in transformed_docs:
                _transformed_doc.metadata[self.chunk_id_key] = _chunk_id
            chunk_vs_ids = self.vectorstore.add_documents(transformed_docs)
            # Inject id of transformed document in the chuck document
            doc.metadata[self.all_transformed_id_key] = ','.join(chunk_vs_ids)
            full_docs.append((_chunk_id, doc))

        if add_to_docstore:
            self.docstore.mset(full_docs)
        if self.parent_transformer:
            if not ids:
                ids: List[str] = []
                # Some docstore can not have an url for the key
                for doc in documents:
                    key = doc.metadata[self.parent_id_key]
                    hash = hashlib.sha256()
                    hash.update(bytes(key, 'utf-8'))
                    ids.append(hash.hexdigest())
            # Save the parent documents  FIXME: flag add_to_docstore
            self.docstore.mset(
                [(parent_id, doc.metadata) for parent_id, doc in zip(ids, documents)])
        if self.parent_transformer:
            #parent_documents_metadata[self.all_transformed_id_key]
            return [doc.metadata[self.parent_id_key] for doc in documents]
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

            parent_documents_metadata:Dict[str,Any] = self.docstore.mget(ids)
            parent_documents_metadata[self.all_transformed_id_key].split(',')
            print(parent_documents)

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
                self.vectorstore.delete(doc.metadata[self.all_transformed_id_key])

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
