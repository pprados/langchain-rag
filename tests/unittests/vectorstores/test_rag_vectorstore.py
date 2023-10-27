import asyncio
import copy
import hashlib
from typing import Iterable, Optional, List, Any, Type, Iterator, Dict, Tuple
from unittest.mock import call

from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore, VST
from langchain.storage import InMemoryStore
from langchain.text_splitter import TokenTextSplitter
from pytest_mock import MockerFixture

from langchain_parent.document_transformers import DocumentTransformers
from langchain_parent.document_transformers.runnable_document_transformer import \
    RunnableGeneratorDocumentTransformer
from langchain_parent.vectorstores import ParentVectorStore


class FakeUUID:
    def __init__(self, prefix):
        self.uuid_count = 0
        self.prefix = prefix

    def __call__(self):
        self.uuid_count += 1
        return f"{self.prefix}{self.uuid_count:0>2}"


def _must_be_called(must_be_called: List[Tuple[List[str], List[Dict[str, Any]]]]):
    calls = []
    for page_contents, metadatas in must_be_called:
        for page_content, metadata in zip(page_contents, metadatas):
            calls.append(
                call([Document(page_content=page_content.upper(), metadata=metadata),
                      Document(page_content=page_content.lower(), metadata=metadata)]))
    return calls


class FakeVectorStore(VectorStore):
    """ Simulation of a vectorstore (without embedding)"""

    # def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
    #     print(f"add_documents({documents=},{kwargs=})")
    #     return super().add_documents(documents=documents,**kwargs)

    def __init__(self):
        self.uuid = FakeUUID(prefix="Fake-VS-")
        self.docs = {}

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        print(f"add_documents({documents=},{kwargs=})")
        uuids = []
        for doc in documents:
            uuid = self.uuid()
            for word in doc.page_content.split(' '):
                word = word.lower()
                l = self.docs.get(word, [])
                l.append(doc)
                self.docs[word] = l
            uuids.append(uuid)
        return uuids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        return [self.uuids() for _ in texts]

    def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        result = []
        for word in query.split(" "):
            word = word.lower()
            if word in self.docs:
                result.extend(self.docs[word])
        return result

    @classmethod
    def from_texts(
            cls: Type[VST],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> VST:
        store = cls()
        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return True


# class UpperDocumentTransformer(RunnableGeneratorDocumentTransformer):
#     def lazy_transform_documents(
#             self, documents: Iterator[Document], **kwargs: Any
#     ) -> Iterator[Document]:
#         yield from (
#             Document(page_content=doc.page_content.upper(),
#                      metadata=copy.deepcopy(doc.metadata))
#             for doc in documents)
#     def transform_documents(
#             self, documents: Sequence[Document], **kwargs: Any
#     ) -> Sequence[Document]:
#         return super().transform_documents(documents,**kwargs)
#
class UpperLazyTransformer(RunnableGeneratorDocumentTransformer):
    def lazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        yield from (Document(page_content=doc.page_content.upper(),
                             metadata=copy.deepcopy(doc.metadata))
                    for doc in documents)

    async def alazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        for doc in documents:
            await asyncio.sleep(0)  # To be sure it's async
            yield Document(
                page_content=doc.page_content.upper(),
                metadata=copy.deepcopy(doc.metadata))


class LowerLazyTransformer(RunnableGeneratorDocumentTransformer):
    def lazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        yield from (Document(page_content=doc.page_content.lower(),
                             metadata=copy.deepcopy(doc.metadata))
                    for doc in documents)

    async def alazy_transform_documents(
            self,
            documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        for doc in documents:
            await asyncio.sleep(0)  # To be sure it's async
            yield Document(
                page_content=doc.page_content.lower(),
                metadata=copy.deepcopy(doc.metadata))


class SplitterWithUniqId(TokenTextSplitter):
    def __init__(self, **values):
        super().__init__(**values)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        documents = super().split_documents(documents)
        for doc in documents:
            doc.metadata[
                "split_id"] = f'{doc.metadata["id"]}-{doc.metadata["start_index"]}'
        return documents


parent_transformer = SplitterWithUniqId(
    chunk_size=1,
    chunk_overlap=0,
    add_start_index=True,
)

chunk_transformer = DocumentTransformers(
    transformers=[
        UpperLazyTransformer(),
        LowerLazyTransformer(),
    ]
)


def test_parent_chunk(mocker: MockerFixture):
    """
    parent_transformer = True
    chunk_transformer = True
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})

    # ----
    ids = vs.add_documents([doc1, doc2])
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 4
    assert ids == [
        hashlib.sha256(str(doc1.metadata[vs.source_id_key]).encode("utf-8")).hexdigest(),
        hashlib.sha256(str(doc2.metadata[vs.source_id_key]).encode("utf-8")).hexdigest(),
    ]
    spy_add_documents.assert_has_calls(_must_be_called([
        (['HELLO'],
         [{'id': 1, 'start_index': 0, 'split_id': '1-0',
           vs.chunk_id_key: 'chunk-01'}, ]),
        ([' WORD'],
         [{'id': 1, 'start_index': 5, 'split_id': '1-5', vs.chunk_id_key: 'chunk-02'}]),
        (['HAPPY'],
         [{'id': 2, 'start_index': 0, 'split_id': '2-0', vs.chunk_id_key: 'chunk-03'}]),
        ([' DAYS'],
         [{'id': 2, 'start_index': 5, 'split_id': '2-5', vs.chunk_id_key: 'chunk-04'}])
    ]))
    spy_delete.assert_called_with(ids=list({f'Fake-VS-0{i}' for i in range(1, 9)}))


def test_parent_chunk_childid(mocker: MockerFixture):
    """
    Sometime, the result of a parent transformation is a list of documents
    with an uniq id. It's not necessery to inject a new one.
    You can set the name of this id with `chunk_id_key`.

    parent_transformer = True
    chunk_transformer = True
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        doc_id_key="id",
        chunk_id_key='split_id',
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})

    # ----
    ids = vs.add_documents([doc1, doc2])
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 4
    assert ids == [
        hashlib.sha256(str(doc1.metadata[vs.source_id_key]).encode("utf-8")).hexdigest(),
        hashlib.sha256(str(doc2.metadata[vs.source_id_key]).encode("utf-8")).hexdigest(),
    ]
    spy_add_documents.assert_has_calls(_must_be_called([
        (['HELLO'], [{'id': 1, 'start_index': 0, 'split_id': '1-0', }]),
        ([' WORD'], [{'id': 1, 'start_index': 5, 'split_id': '1-5', }]),
        (['HAPPY'], [{'id': 2, 'start_index': 0, 'split_id': '2-0', }]),
        ([' DAYS'], [{'id': 2, 'start_index': 5, 'split_id': '2-5', }])
    ]))
    spy_delete.assert_called_with(list({f'Fake-VS-0{i}' for i in range(1, 9)}))


def test_parent_chunk_ids(mocker: MockerFixture):
    """
    parent_transformer = True
    chunk_transformer = True
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    force_ids = [fake_uuid(), fake_uuid()]

    # ----
    ids = vs.add_documents([doc1, doc2],ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 4
    assert ids == force_ids
    spy_add_documents.assert_has_calls(_must_be_called([
        (['HELLO'],
         [{'id': 1, 'start_index': 0, 'split_id': '1-0',
           vs.chunk_id_key: 'chunk-01'}, ]),
        ([' WORD'],
         [{'id': 1, 'start_index': 5, 'split_id': '1-5', vs.chunk_id_key: 'chunk-02'}]),
        (['HAPPY'],
         [{'id': 2, 'start_index': 0, 'split_id': '2-0', vs.chunk_id_key: 'chunk-03'}]),
        ([' DAYS'],
         [{'id': 2, 'start_index': 5, 'split_id': '2-5', vs.chunk_id_key: 'chunk-04'}])
    ]))
    spy_delete.assert_called_with(ids=list({f'Fake-VS-0{i}' for i in range(1, 9)}))

def test_parent_ids(mocker: MockerFixture):
    """
    parent_transformer = True
    chunk_transformer = False
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})

    # ----
    force_ids = [fake_uuid(), fake_uuid()]
    ids = vs.add_documents(documents=[doc1, doc2], ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 4
    assert ids == force_ids
    spy_add_documents.assert_has_calls(_must_be_called([
        (['HELLO'],
         [{'id': 1, 'start_index': 0, 'split_id': '1-0', vs.chunk_id_key: 'chunk-01'}]),
        ([' WORD'],
         [{'id': 1, 'start_index': 5, 'split_id': '1-5', vs.chunk_id_key: 'chunk-02'}]),
        (['HAPPY'],
         [{'id': 2, 'start_index': 0, 'split_id': '2-0', vs.chunk_id_key: 'chunk-03'}]),
        ([' DAYS'],
         [{'id': 2, 'start_index': 5, 'split_id': '2-5', vs.chunk_id_key: 'chunk-04'}])
    ]))
    spy_delete.assert_called_with(list({f'Fake-VS-0{i}' for i in range(1, 9)}))


# %%
def test_chunk(mocker: MockerFixture):
    """
    parent_transformer = False
    chunk_transformer = True
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    # ----

    ids = vs.add_documents(documents=split_docs)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 4
    spy_add_documents.assert_has_calls(_must_be_called([
        (['HELLO'],
         [{'id': 1, 'start_index': 0, 'split_id': '1-0', vs.chunk_id_key: 'chunk-01'}]),
        ([' WORD'],
         [{'id': 1, 'start_index': 5, 'split_id': '1-5', vs.chunk_id_key: 'chunk-02'}]),
        (['HAPPY'],
         [{'id': 2, 'start_index': 0, 'split_id': '2-0', vs.chunk_id_key: 'chunk-03'}]),
        ([' DAYS'],
         [{'id': 2, 'start_index': 5, 'split_id': '2-5', vs.chunk_id_key: 'chunk-04'}])
    ]))
    spy_delete.assert_called_with(list({f'Fake-VS-0{i}' for i in range(1, 9)}))


def test_chunk_ids(mocker: MockerFixture):
    """
    parent_transformer = False
    chunk_transformer = True
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=chunk_transformer,
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    force_ids = [fake_uuid() for _ in range(0, len(split_docs))]
    # ----
    ids = vs.add_documents(documents=split_docs, ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 4
    assert ids == force_ids
    spy_add_documents.assert_has_calls(_must_be_called([
        (['HELLO'],
         [{'id': 1, 'start_index': 0, 'split_id': '1-0',
           vs.chunk_id_key: 'persistance-01'}]),
        ([' WORD'],
         [{'id': 1, 'start_index': 5, 'split_id': '1-5',
           vs.chunk_id_key: 'persistance-02'}]),
        (['HAPPY'],
         [{'id': 2, 'start_index': 0, 'split_id': '2-0',
           vs.chunk_id_key: 'persistance-03'}]),
        ([' DAYS'],
         [{'id': 2, 'start_index': 5, 'split_id': '2-5',
           vs.chunk_id_key: 'persistance-04'}])
    ]))
    spy_delete.assert_called_with(list({f'Fake-VS-0{i}' for i in range(1, 9)}))


# %%
def test_parent(mocker: MockerFixture):
    """
    parent_transformer = True
    chunk_transformer = False
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    # ----
    ids = vs.add_documents(documents=[doc1, doc2])
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 1
    assert ids == [
        hashlib.sha256(str(doc1.metadata[vs.source_id_key]).encode("utf-8")).hexdigest(),
        hashlib.sha256(str(doc2.metadata[vs.source_id_key]).encode("utf-8")).hexdigest(),
    ]
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(page_content='Hello',
                             metadata={'id': 1, 'start_index': 0,
                                       'split_id': '1-0'}),
                    Document(page_content=' word',
                             metadata={'id': 1, 'start_index': 5,
                                       'split_id': '1-5'}),
                    Document(page_content='Happy',
                             metadata={'id': 2, 'start_index': 0,
                                       'split_id': '2-0'}),
                    Document(page_content=' days',
                             metadata={'id': 2, 'start_index': 5,
                                       'split_id': '2-5'}),
                ],
                ids=[f'chunk-0{i}' for i in range(1, 5)]
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f'chunk-0{i}' for i in range(1, 5)])


def test_parent_ids(mocker: MockerFixture):
    """
    parent_transformer = True
    chunk_transformer = False
    ids = Yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=parent_transformer,
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    docs = [doc1, doc2]
    force_ids = [fake_uuid() for _ in range(0, len(docs))]
    # ----
    ids = vs.add_documents(documents=docs, ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 1
    assert ids == force_ids
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(page_content='Hello',
                             metadata={'id': 1, 'start_index': 0,
                                       'split_id': '1-0'}),
                    Document(page_content=' word',
                             metadata={'id': 1, 'start_index': 5,
                                       'split_id': '1-5'}),
                    Document(page_content='Happy',
                             metadata={'id': 2, 'start_index': 0,
                                       'split_id': '2-0'}),
                    Document(page_content=' days',
                             metadata={'id': 2, 'start_index': 5,
                                       'split_id': '2-5'}),
                ],
                ids=[f'chunk-0{i}' for i in range(1, 5)]
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f'chunk-0{i}' for i in range(1, 5)])


# %%
def test_nothing(mocker: MockerFixture):
    """
    parent_transformer = False
    chunk_transformer = False
    ids = No
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    # ----

    # RunnableGenerator(
    #     transform=partial(_transform_documents_generator,
    #                       transformers=self.transformers)
    # ).transform([doc1])

    ids = vs.add_documents(documents=split_docs)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 1
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(page_content='Hello',
                             metadata={'id': 1, 'start_index': 0,
                                       'split_id': '1-0'}),
                    Document(page_content=' word',
                             metadata={'id': 1, 'start_index': 5,
                                       'split_id': '1-5'}),
                    Document(page_content='Happy',
                             metadata={'id': 2, 'start_index': 0,
                                       'split_id': '2-0'}),
                    Document(page_content=' days',
                             metadata={'id': 2, 'start_index': 5,
                                       'split_id': '2-5'}),
                ],
                ids=[f'chunk-0{i}' for i in range(1, 5)]
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f'chunk-0{i}' for i in range(1, 5)])


def test_nothing_ids(mocker: MockerFixture):
    """
    parent_transformer = False
    chunk_transformer = False
    ids = yes
    """
    fake_vs = FakeVectorStore()
    docstore = InMemoryStore()
    spy_add_documents = mocker.spy(fake_vs, 'add_documents')
    spy_delete = mocker.spy(fake_vs, 'delete')
    mock_uuid4 = mocker.patch("uuid.uuid4")
    mock_uuid4.side_effect = FakeUUID("chunk-")

    fake_uuid = FakeUUID(prefix="persistance-")
    vs = ParentVectorStore(
        vectorstore=fake_vs,
        parent_transformer=None,  # No parent transformer.
        chunk_transformer=None,  # No child transformer
        docstore=docstore,
        doc_id_key="id",
    )
    doc1 = Document(page_content="Hello word", metadata={"id": 1})
    doc2 = Document(page_content="Happy days", metadata={"id": 2})
    split_docs = parent_transformer.split_documents([doc1, doc2])
    force_ids = [fake_uuid() for _ in range(0, len(split_docs))]
    # ----
    ids = vs.add_documents(documents=split_docs, ids=force_ids)
    result = vs.as_retriever().get_relevant_documents(doc1.page_content)
    vs.delete(ids)
    # ----
    assert result[0].page_content == "Hello"
    assert result[1].page_content == " word"
    assert spy_add_documents.call_count == 1
    assert ids == force_ids
    spy_add_documents.assert_has_calls(
        [
            call(
                documents=[
                    Document(page_content='Hello',
                             metadata={'id': 1, 'start_index': 0,
                                       'split_id': '1-0'}),
                    Document(page_content=' word',
                             metadata={'id': 1, 'start_index': 5,
                                       'split_id': '1-5'}),
                    Document(page_content='Happy',
                             metadata={'id': 2, 'start_index': 0,
                                       'split_id': '2-0'}),
                    Document(page_content=' days',
                             metadata={'id': 2, 'start_index': 5,
                                       'split_id': '2-5'}),
                ],
                ids=[f'persistance-0{i}' for i in range(1, 5)]
            )
        ]
    )
    spy_delete.assert_called_with(ids=[f'persistance-0{i}' for i in range(1, 5)])
