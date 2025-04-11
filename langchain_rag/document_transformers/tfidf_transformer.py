from copy import copy
from typing import Any, Iterator

import nltk
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from nltk import WordNetLemmatizer, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords

from langchain_rag.document_transformers import LazyDocumentTransformer

def _ensure_nltk_resources(resources):
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1])


class LemmatizeDocumentTransformer(LazyDocumentTransformer):

    def __init__(self,
                 language: str
                 ):
        self.language = language
        _ensure_nltk_resources([
            'corpora/wordnet',
            'corpora/stopwords'
        ])

    def _alazy_transform_documents(
            self, documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        yield from self.lazy_transform_documents(documents, **kwargs)

    def lazy_transform_documents(
            self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        for doc in documents:
            text = doc.page_content

            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words(self.language))
            words = word_tokenize(text.lower(), language=self.language)
            stemmed_text = ' '.join(lemmatizer.lemmatize(word) for word in words if
                                    word.isalpha() and word not in stop_words)

            yield Document(
                page_content=stemmed_text,
                metadata=copy(doc.metadata)
            )


class StemmerDocumentTransformer(LazyDocumentTransformer):

    def __init__(self,
                 *,
                 language: str,
        ignore_stopwords:bool = False
                 ):
        self.language = language
        self.ignore_stopwords = ignore_stopwords
        _ensure_nltk_resources([
            'corpora/wordnet',
            'corpora/stopwords'
        ])

    def _alazy_transform_documents(
            self, documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        yield from self.lazy_transform_documents(documents, **kwargs)

    def lazy_transform_documents(
            self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        for doc in documents:
            text = doc.page_content

            stemmer = SnowballStemmer(language=self.language,
                                      ignore_stopwords=self.ignore_stopwords)
            stop_words = set(stopwords.words(self.language))
            words = word_tokenize(text.lower(), language=self.language)
            stemmed_text = ' '.join(stemmer.stem(word) for word in words
                                    if word.isalpha() and word not in stop_words
                                    )

            yield Document(
                page_content=stemmed_text,
                metadata=copy(doc.metadata)
            )


class TFIDFTransformer(LazyDocumentTransformer):
    """
    Use like this.
    ```python
    tfidf_transformer = TFIDFTransformer()
    pipeline=DocumentTransformerPipeline(
        transformers=
        [
            LemmatizeDocumentTransformer(language="english"),
            # or StemmerDocumentTransformer(language="english"),
            tfidf_transformer
        ]).transform_documents(
        [
            Document(page_content="saying hello world")
        ])
    pipeline.transform_documents(documents)
    tfidf_transformer.tfidf_retriever.save_local(".","tfidf.pkl")

    tfidf_retriever=TFIDFRetriever.load_local(".","tfidf.pkl")
    ```
    """

    def __init__(self,
                 tfidf_params:dict=None
                 ):
        self.tfidf_retriever = None
        self.tfidf_params = tfidf_params

    def _alazy_transform_documents(
            self, documents: Iterator[Document],
            **kwargs: Any
    ) -> Iterator[Document]:
        yield from self.lazy_transform_documents(documents, **kwargs)

    def lazy_transform_documents(
            self, documents: Iterator[Document], **kwargs: Any
    ) -> Iterator[Document]:
        documents = list(documents)  # Yes, it's not lazy :-(
        # TODO: TFIDFRetriever est un fit global, pour identifier le vocabulaire. Pas de MAJ
        # TODO TFIDFRetriever(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, **kwargs)
        # TODO: alternative HashingVectorizer. Possibilité d'ajout et delete (voir chatgpt)
        self.tfidf_retriever = TFIDFRetriever.from_documents(
            documents,
            tfidf_params=self.tfidf_params)
        return iter([])
