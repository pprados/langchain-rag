Langchain-RAG
=============

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pprados/langchain-rag?quickstart=1)

> Note: A [pull-request](https://github.com/langchain-ai/langchain/pull/13910) with this code was proposed to langchain.

When splitting documents for retrieval, there are often conflicting desires:

1. You may want to keep documents small, ensuring that their embeddings accurately represent their meaning. If they become too long, the embeddings can lose their meaning.
2. You also want to maintain documents long enough to retain the context of each chunk.

When you have a lot of documents, and therefore a lot of pieces, it's likely that dozens of pieces have a distance close to the question. Taking only the top 4 is not a good idea. The answer may lie in the 6 or 7 tracks. How can we improve the match between the question and a fragment? By preparing several versions of the fragment, each with an embedding. In this way, one of the versions can be closer to the question than the original fragment. This version is stripped of context. But the context is still needed to answer the question correctly. One strategy consists of breaking down each fragment into different versions, but using the retriever to return to the original fragment. 

The `RAGVectorStore` strikes a balance by splitting and storing small chunks and different variations of data. During retrieval, it initially retrieves the small chunks but then looks up the parent IDs for those chunks and returns the larger documents.

The challenge lies in correctly managing the lifecycle of the three levels of documents:
- Original documents
- Chunks extracted from the original documents
- Transformations of chunks to generate more vectors for improved retrieval

The `RAGVectorStore`, in combination with other components, is designed to address this challenge.

# Demo
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pprados/langchain-rag/blob/master/docs/integrations/vectorstores/rag_vectorstore.ipynb)

Or :
- `poetry run python -m ipykernel install --user --name langchain-rag`
- `jupyter lab`

# Tips
`poetry run python -m ipykernel install --user --name langchain-parent`