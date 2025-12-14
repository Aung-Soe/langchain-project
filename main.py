import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """Use the context below to answer the question.
    Limit answer to maximum 3 sentences and be concise.
    If you don’t know, say you don’t know.

    Context:
    {context}

    Question:
    {input}
    """
)

combine_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(
    retriever,
    combine_chain
)

result = retrieval_chain.invoke({"input": "What is Pinecone?"})
print(result["answer"])