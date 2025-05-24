from fastapi import FastAPI
from pydantic import BaseModel
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import io
from contextlib import redirect_stdout

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "nlcs-dataset"

# Initialize Pinecone & LangChain
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Agent setup
vectorstore_info = VectorStoreInfo(
    name="AI for Land Record and Management",
    description="Knowledge base for land ownership, cadastral mapping, land tax, GIS, and related government processes.",
    vectorstore=docsearch,
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(query: Query):
    try:
        # Capture verbose agent output (Thought, Action, etc.)
        f = io.StringIO()
        with redirect_stdout(f):
            answer = agent_executor.run(query.question)
        trace_output = f.getvalue()

        # Also extract top relevant sources (URLs + page)
        retriever = docsearch.as_retriever(search_kwargs={"k": 3})
        source_docs = retriever.get_relevant_documents(query.question)
        sources = [
            {
                "url": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            }
            for doc in source_docs
        ]

        return {
            "answer": answer,
            "trace": trace_output,
            "sources": sources
        }
    except Exception as e:
        return {"error": str(e)}
