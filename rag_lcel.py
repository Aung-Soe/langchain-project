"""
pip uninstall -y langchain langchain-classic
pip install -U \
  langchain==0.2.* \
  langchain-core==0.2.* \
  langchain-openai==0.1.* \
  langchain-pinecone==0.1.* \
  pinecone-client==3.*
pip install -U python-dotenv
"""
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# --- ENV VARS REQUIRED ---
# OPENAI_API_KEY
# PINECONE_API_KEY
# PINECONE_ENVIRONMENT
# INDEX_NAME

def main():
    # LLM + Embeddings
    llm = ChatOpenAI(
        model="gpt-4o-mini",   # or gpt-4.1 / gpt-4o
        temperature=0
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Pinecone Vector Store
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # Prompt (LCEL-compatible)
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
        Use ONLY the context below to answer the question.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {input}
        """
    )

    # Document → Answer chain
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # Retrieval → Answer (LCEL)
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_documents_chain=document_chain
    )

    # Invoke
    query = "What is Pinecone in machine learning?"
    result = rag_chain.invoke({"input": query})

    print("\nAnswer:\n", result["answer"])

if __name__ == "__main__":
    main()

