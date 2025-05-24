from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os

# ENV variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "nlcs-dataset"

# Init Pinecone + embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
retriever = docsearch.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: Query):
    try:
        response = qa_chain(data.question)
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            }
            for doc in response.get("source_documents", [])
        ]
        return {
            "answer": response["result"],
            "sources": sources
        }
    except Exception as e:
        return {"error": str(e)}
