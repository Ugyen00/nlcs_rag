# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# from langchain_anthropic import ChatAnthropic
# from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# import os
# import io
# from contextlib import redirect_stdout
# from fastapi.middleware.cors import CORSMiddleware
# import random
# from dotenv import load_dotenv
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pinecone import Pinecone, ServerlessSpec
# import time
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# #memory
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# load_dotenv()

# # Env vars
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
# INDEX_NAME = "nlcs-dataset"

# # Pinecone + Embeddings
# pc = Pinecone(api_key=PINECONE_API_KEY)
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

# # Claude LLM (Anthropic)
# llm = ChatAnthropic(
#     anthropic_api_key=ANTHROPIC_API_KEY,
#     model="claude-3-sonnet-20240229",
#     temperature=0,
#     max_tokens=1024
# )

# # Agent setup
# vectorstore_info = VectorStoreInfo(
#     name="AI for Land Record and Management",
#     description="Knowledge base for land ownership, cadastral mapping, land tax, GIS, and related government processes.",
#     vectorstore=docsearch,
# )

# retriever = docsearch.as_retriever(search_kwargs={"k": 5})

# # toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
# toolkit = VectorStoreToolkit(
#     vectorstore_info=VectorStoreInfo(
#         name="AI for Land Record and Management",
#         description="Knowledge base...",
#         vectorstore=docsearch,
#     ),
#     llm=llm,
#     retriever=retriever
# )

# agent_executor = create_vectorstore_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     handle_parsing_errors=True
# )

# # FastAPI app
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "https://agay.vercel.app"
#     ],
#     allow_credentials=True,
#     allow_methods=["POST"],
#     allow_headers=["*"],
# )

# # Google Drive credentials
# SERVICE_ACCOUNT_FILE = "agay-462121-24ecbff56947.json"
# GDRIVE_FOLDER_ID = "1OfoLfA6H43Jpnh3Rt3GH0ZpV96NB1HCL"

# drive_creds = service_account.Credentials.from_service_account_file(
#     SERVICE_ACCOUNT_FILE,
#     scopes=["https://www.googleapis.com/auth/drive"],
# )
# drive_service = build("drive", "v3", credentials=drive_creds)

# class Query(BaseModel):
#     question: str
#     session_id: str

# session_memories = {}

# @app.post("/ask")
# async def ask_agent(query: Query):
#     try:
#         user_input = query.question.lower().strip()

#         greeting_variants = [
#             "👋 Hello! How can I help you with land records today?",
#             "Hey there! Need help with a land query? 😊",
#             "Hi! Ask me anything about land ownership, location, or details.",
#             "Kuzuzangpo! I'm here to assist you with land records. 🙏",
#             "Greetings! What would you like to know about land holdings?",
#             "👋 Hi again! I'm ready to fetch land info for you.",
#             "Hello! Want to know about land types, locations, or owners?"
#         ]

#         if user_input in ["hi", "hello", "hey", "yo", "greetings", "kuzu", "kuzuzangpo"]:
#             return {
#                 "answer": random.choice(greeting_variants),
#                 "trace": "Friendly greeting detected. No agent call made."
#             }

#         if query.session_id not in session_memories:
#             session_memories[query.session_id] = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 return_messages=True
#             )

#         memory = session_memories[query.session_id]

#         rag_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=docsearch.as_retriever(),
#             memory=memory,
#             verbose=True
#         )

#         f = io.StringIO()
#         with redirect_stdout(f):
#             try:
#                 answer = rag_chain.run(query.question)
#             except Exception as parse_error:
#                 answer = str(parse_error)

#         trace_output = f.getvalue()

#         if (
#             "I don't have enough information" in answer
#             or "I'm afraid" in answer
#             or "OUTPUT_PARSING_FAILURE" in answer
#         ):
#             answer = "The agent couldn’t find enough information to answer that question confidently."

#         return {
#             "answer": answer,
#             "trace": trace_output
#         }

#     except Exception as e:
#         return {"error": str(e)}


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         file_content = await file.read()
#         media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype=file.content_type)

#         file_metadata = {
#             "name": file.filename,
#             "parents": [GDRIVE_FOLDER_ID],
#         }

#         uploaded_file = (
#             drive_service.files()
#             .create(body=file_metadata, media_body=media, fields="id")
#             .execute()
#         )

#         return {"message": "✅ File uploaded to Google Drive!", "file_id": uploaded_file["id"]}

#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/train")
# def train_from_drive():
#     try:
#         # List files from Drive folder
#         query = f"'{GDRIVE_FOLDER_ID}' in parents and mimeType='application/pdf'"
#         results = drive_service.files().list(q=query, fields="files(id, name)").execute()
#         files = results.get("files", [])

#         os.makedirs("/tmp/nlcs", exist_ok=True)
#         pdf_paths = []

#         for file in files:
#             file_id = file["id"]
#             file_name = file["name"]
#             file_path = f"/tmp/nlcs/{file_name}"

#             request = drive_service.files().get_media(fileId=file_id)
#             with open(file_path, "wb") as f:
#                 downloader = MediaIoBaseDownload(f, request)
#                 done = False
#                 while not done:
#                     _, done = downloader.next_chunk()

#             pdf_paths.append(file_path)

#         # Load and embed
#         loader = DirectoryLoader("/tmp/nlcs", glob="*.pdf", loader_cls=PyPDFLoader)
#         documents = loader.load()
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = splitter.split_documents(documents)

#         PineconeVectorStore.from_documents(
#             documents=chunks,
#             embedding=embeddings,
#             index_name=INDEX_NAME
#         )
        
#         for i, chunk in enumerate(chunks[:5]):
#             print(f"Chunk {i + 1} preview:\n{chunk.page_content}\n")
            
#         print(f"✅ Embedded {len(chunks)} chunks to Pinecone")

#         return {"message": "✅ Training complete from Google Drive folder. Data embedded to Pinecone."}

#     except Exception as e:
#         return {"error": str(e)}

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
import io
from contextlib import redirect_stdout
from fastapi.middleware.cors import CORSMiddleware
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import json

# Memory for agent conversations - FIXED VERSION
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

# Environment variables

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
INDEX_NAME = "nlcs-dataset"

# Pinecone + Embeddings setup
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

# Claude LLM (Updated to newer model)
llm = ChatAnthropic(
    anthropic_api_key=ANTHROPIC_API_KEY,
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=2048
)

# Agent setup with proper configuration
vectorstore_info = VectorStoreInfo(
    name="AI for Land Record and Management",
    description="Comprehensive knowledge base for land ownership, cadastral mapping, land tax, GIS, property records, and related government processes in Bhutan.",
    vectorstore=docsearch,
)

# Create retriever with better search parameters
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# Create toolkit
toolkit = VectorStoreToolkit(
    vectorstore_info=vectorstore_info,
    llm=llm
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
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Google Drive credentials
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")


# Initialize Google Drive service
try:
    drive_creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive_service = build("drive", "v3", credentials=drive_creds)
except Exception as e:
    print(f"Warning: Google Drive setup failed: {e}")
    drive_service = None

class Query(BaseModel):
    question: str
    session_id: str

# FIXED: Better session memory storage with conversation context
session_conversations = {}

def get_or_create_conversation(session_id: str) -> list:
    """Get existing conversation history for session or create new one"""
    if session_id not in session_conversations:
        session_conversations[session_id] = []
    return session_conversations[session_id]

def add_to_conversation(session_id: str, user_message: str, ai_response: str):
    """Add exchange to conversation history"""
    conversation = get_or_create_conversation(session_id)
    conversation.append({"role": "user", "content": user_message})
    conversation.append({"role": "assistant", "content": ai_response})
    
    # Keep only last 20 messages (10 exchanges) to avoid token limits
    if len(conversation) > 20:
        session_conversations[session_id] = conversation[-20:]

def format_conversation_context(session_id: str) -> str:
    """Format conversation history for context"""
    conversation = get_or_create_conversation(session_id)
    if not conversation:
        return ""
    
    context_parts = []
    for msg in conversation[-10:]:  # Last 5 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(context_parts)

def create_agent_with_context():
    """Create agent executor - simplified version"""
    return create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

@app.get("/")
async def root():
    return {"message": "Land Records RAG Agent API with Memory is running!"}

@app.post("/ask")
async def ask_agent(query: Query):
    try:
        user_input = query.question.lower().strip()

        # Greeting responses
        greeting_variants = [
            "👋 Hello! How can I help you with land records today?",
            "Hey there! Need help with a land query? 😊",
            "Hi! Ask me anything about land ownership, location, or details.",
            "Kuzuzangpo! I'm here to assist you with land records. 🙏",
            "Greetings! What would you like to know about land holdings?",
            "👋 Hi again! I'm ready to fetch land info for you.",
            "Hello! Want to know about land types, locations, or owners?"
        ]

        # Handle greetings
        if user_input in ["hi", "hello", "hey", "yo", "greetings", "kuzu", "kuzuzangpo"]:
            greeting_response = random.choice(greeting_variants)
            add_to_conversation(query.session_id, query.question, greeting_response)
            
            return {
                "answer": greeting_response,
                "trace": "Friendly greeting detected. Response stored in conversation history.",
                "session_id": query.session_id,
                "conversation_length": len(get_or_create_conversation(query.session_id))
            }

        # Get conversation context
        conversation_context = format_conversation_context(query.session_id)
        
        # Create agent
        agent_executor = create_agent_with_context()

        # Capture stdout for debugging
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                # FIXED: Create enhanced prompt with proper conversation context
                context_prompt = ""
                if conversation_context:
                    context_prompt = f"""
                    IMPORTANT CONVERSATION CONTEXT (Previous messages in this session):
                    {conversation_context}

                    CURRENT CONTEXT: The above conversation history is from the same session. When the user refers to "she", "he", "it", "this person", "that land", etc., check the conversation history above to understand what they're referring to.
                    """

                    enhanced_prompt = f"""{context_prompt}
                    You are a helpful assistant specializing in land records and property management in Bhutan.

                    Current User Question: {query.question}

                    Instructions:
                    1. FIRST check the conversation context above to understand any references (like "she", "he", "this person", "that land")
                    2. If the user is asking a follow-up question, use the conversation history to understand the context
                    3. Search the knowledge base thoroughly for relevant information
                    4. Provide a comprehensive answer based on available documents and conversation context
                    5. If you cannot find specific information, clearly state what information is missing

                    Always prioritize accuracy and cite relevant information from the documents when possible.
                    """
                
                # Invoke the agent with the enhanced prompt
                result = agent_executor.invoke({
                    "input": enhanced_prompt
                })
                
                # Extract the answer from the result
                answer = result.get("output", "No response generated")
                
            except Exception as parse_error:
                print(f"Agent execution error: {parse_error}")
                # FIXED: Better fallback with conversation context
                try:
                    docs = retriever.get_relevant_documents(query.question)
                    if docs:
                        context = "\n\n".join([doc.page_content for doc in docs[:3]])
                        
                        fallback_prompt = f"""Based on this context about land records:
                        
                        {context}

                        Conversation History:
                        {conversation_context}

                        Current Question: {query.question}

                        Please answer the question considering both the document context and the conversation history. If the user is referring to someone or something mentioned earlier, use that context to provide a relevant answer."""
                                                
                        # Direct LLM call as fallback
                        llm_response = llm.invoke(fallback_prompt)
                        answer = llm_response.content
                        
                    else:
                        answer = "I couldn't find relevant information in the knowledge base to answer your question."
                        
                except Exception as fallback_error:
                    answer = f"I encountered an error while processing your question: {str(fallback_error)}"

        # Add to conversation history
        add_to_conversation(query.session_id, query.question, answer)

        # Get trace output
        trace_output = f.getvalue()
        
        # Clean up trace output (remove ANSI codes)
        clean_trace = trace_output.replace("\u001b[0m", "").replace("\u001b[1m", "").replace("\u001b[32;1m", "").replace("\u001b[33;1m", "")

        # Handle common error patterns
        if (
            "I don't have enough information" in answer
            or "I'm afraid" in answer
            or "OUTPUT_PARSING_FAILURE" in answer
            or not answer.strip()
        ):
            answer = "I couldn't find enough specific information in the knowledge base to answer that question confidently. Could you please provide more details or rephrase your question?"
            
            # Update conversation with refined answer
            if session_conversations.get(query.session_id):
                session_conversations[query.session_id][-1]["content"] = answer

        return {
            "answer": answer,
            "trace": clean_trace if clean_trace else "Agent executed successfully with conversation context",
            "session_id": query.session_id,
            "conversation_length": len(get_or_create_conversation(query.session_id))
        }

    except Exception as e:
        print(f"Unexpected error in ask_agent: {e}")
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "answer": "I'm sorry, I encountered an error while processing your request. Please try again."
        }

@app.post("/clear_memory")
async def clear_session_memory(session_id: str):
    """Clear conversation history for a specific session"""
    try:
        if session_id in session_conversations:
            del session_conversations[session_id]
            return {"message": f"Conversation history cleared for session: {session_id}"}
        else:
            return {"message": f"No conversation history found for session: {session_id}"}
    except Exception as e:
        return {"error": f"Failed to clear conversation history: {str(e)}"}

@app.get("/memory_info/{session_id}")
async def get_memory_info(session_id: str):
    """Get conversation information for a session"""
    try:
        conversation = get_or_create_conversation(session_id)
        return {
            "session_id": session_id,
            "message_count": len(conversation),
            "last_messages": conversation[-6:] if conversation else []  # Last 3 exchanges
        }
    except Exception as e:
        return {"error": f"Failed to get conversation info: {str(e)}"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not drive_service:
            return {"error": "Google Drive service is not available"}
            
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return {"error": "Only PDF files are supported"}
            
        file_content = await file.read()
        media = MediaIoBaseUpload(io.BytesIO(file_content), mimetype=file.content_type)

        file_metadata = {
            "name": file.filename,
            "parents": [GDRIVE_FOLDER_ID],
        }

        uploaded_file = (
            drive_service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        return {
            "message": "✅ File uploaded to Google Drive successfully!",
            "file_id": uploaded_file["id"],
            "filename": file.filename
        }

    except Exception as e:
        print(f"Upload error: {e}")
        return {"error": f"Upload failed: {str(e)}"}

@app.post("/train")
def train_from_drive():
    try:
        if not drive_service:
            return {"error": "Google Drive service is not available"}
            
        # List PDF files from Drive folder
        query = f"'{GDRIVE_FOLDER_ID}' in parents and mimeType='application/pdf'"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])
        
        if not files:
            return {"error": "No PDF files found in the Google Drive folder"}

        # Create temporary directory
        temp_dir = "/tmp/nlcs"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download files from Google Drive
        downloaded_files = []
        for file in files:
            try:
                file_id = file["id"]
                file_name = file["name"]
                file_path = f"{temp_dir}/{file_name}"

                request = drive_service.files().get_media(fileId=file_id)
                with open(file_path, "wb") as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()

                downloaded_files.append(file_path)
                print(f"Downloaded: {file_name}")
                
            except Exception as e:
                print(f"Error downloading {file.get('name', 'unknown')}: {e}")
                continue

        if not downloaded_files:
            return {"error": "No files were successfully downloaded"}

        # Load and process documents
        loader = DirectoryLoader(temp_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            return {"error": "No documents were loaded from the PDF files"}

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            return {"error": "No text chunks were created from the documents"}

        # Preview chunks (for debugging)
        print(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i + 1} preview (first 200 chars):\n{chunk.page_content[:200]}...\n")

        # Add to Pinecone
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        
        # Clean up temporary files
        for file_path in downloaded_files:
            try:
                os.remove(file_path)
            except:
                pass

        print(f"✅ Successfully embedded {len(chunks)} chunks to Pinecone")

        return {
            "message": f"✅ Training completed successfully! Processed {len(files)} files and embedded {len(chunks)} chunks to Pinecone.",
            "files_processed": len(files),
            "chunks_created": len(chunks)
        }

    except Exception as e:
        print(f"Training error: {e}")
        return {"error": f"Training failed: {str(e)}"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent_ready": True,
        "vectorstore_ready": docsearch is not None,
        "drive_service_ready": drive_service is not None,
        "active_sessions": len(session_conversations)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)