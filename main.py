from fastapi import FastAPI
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
import io
from contextlib import redirect_stdout
from fastapi.middleware.cors import CORSMiddleware

# Env vars
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # still needed for embeddings
INDEX_NAME = "nlcs-dataset"

# Pinecone + Embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

# Claude LLM (Anthropic)
llm = ChatAnthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-3-sonnet-20240229",  # or claude-3-haiku / opus if you have access
    # model="Claude Haiku 3",
    temperature=0,
    max_tokens=1024
)

# Agent setup
vectorstore_info = VectorStoreInfo(
    name="AI for Land Record and Management",
    description="Knowledge base for land ownership, cadastral mapping, land tax, GIS, and related government processes.",
    vectorstore=docsearch,
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True
)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://agay.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# @app.post("/ask")
# async def ask_agent(query: Query):
#     try:
#         # Capture stdout for trace
#         f = io.StringIO()
#         with redirect_stdout(f):
#             answer = agent_executor.run(query.question)
#         trace_output = f.getvalue()

#     # Fallback if LLM says "I don't know"
#         if "I don't have enough information" in answer or "I'm afraid" in answer:
#             answer = "The agent couldn’t find enough information to answer that question."

#         return {
#             "answer": answer,
#             "trace": trace_output
#         }
#     except Exception as e:
#         return {"error": str(e)}

@app.post("/ask")
async def ask_agent(query: Query):
    try:
        # Capture trace logs
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                answer = agent_executor.run(query.question)
            except Exception as parse_error:
                # Return Claude's raw output if parsing fails
                answer = str(parse_error)
        trace_output = f.getvalue()

        # Fallback if Claude couldn’t answer properly
        if (
            "I don't have enough information" in answer
            or "I'm afraid" in answer
            or "OUTPUT_PARSING_FAILURE" in answer
        ):
            answer = "The agent couldn’t find enough information to answer that question confidently."

        return {
            "answer": answer,
            "trace": trace_output
        }

    except Exception as e:
        return {"error": str(e)}

