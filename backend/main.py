from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from groq import Groq
import os
from dotenv import load_dotenv
from supabase import create_client
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="DevVibe AI Chatbot")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://devvibe-chatbot.vercel.app",
        "https://devvibe-chatbot-ittogm7sv-dev-vibe2.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "DevVibe AI Chatbot Backend is running!"}


def retrieve_context(question: str):
    query_embedding = list(
        embedding_model.embed([question])
    )[0].tolist()

    matches = supabase.rpc(
        "match_documents_fastembed",
        {
            "query_embedding": query_embedding,
            "match_count": 5,
        },
    ).execute()

    return matches.data

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    documents = retrieve_context(request.message)

    if not documents:
        context = "No relevant information was found in the DevVibe website data."
    else:
        context = "\n\n".join(
            f"Page: {doc['title']}\n{doc['content']}"
            for doc in documents
        )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the DevVibe website assistant. "
                    "Answer using only the provided website context. "
                    "If the answer is unavailable, say you couldn't "
                    "find that information on the website. "
                    "Never invent services, prices, or contact details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Website context:\n{context}\n\n"
                    f"User question: {request.message}"
                ),
            },
        ],
    )

    return {"response": response.choices[0].message.content}