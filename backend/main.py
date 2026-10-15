import os
import re

# Configure model cache directories
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed_cache"

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding
from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GROQ_FALLBACK_MODELS",
        "openai/gpt-oss-120b,qwen/qwen3.8-27b,allam-2-7b",
    ).split(",")
    if model.strip()
]

BUSY_MESSAGE = (
    "I'm getting a lot of questions right now and have hit a temporary "
    "usage limit. Please give me a few minutes and ask again."
)
CONNECTION_MESSAGE = (
    "I'm having trouble reaching my knowledge service right now. "
    "Please check your connection and try again in a moment."
)

if not SUPABASE_URL or not SUPABASE_KEY or not GROQ_API_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL, SUPABASE_KEY, or GROQ_API_KEY "
        "in the environment configuration."
    )

# Initialize existing services
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="DevVibe AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://devvibe-chatbot.vercel.app",
        "https://devvibe-chatbot-ittogm7sv-dev-vibe2.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)

import json
from pathlib import Path


def _norm(text: str) -> str:
    return re.sub(
        r"\s+", " ",
        text.lower().replace("’", "'").replace("‑", "-"),
    ).strip()


with open(Path(__file__).parent / "project_links.json", encoding="utf-8") as f:
    PROJECT_LINKS = {_norm(k): (k, v) for k, v in json.load(f).items()}


def find_project_links(content: str) -> list[tuple[str, str]]:
    text = _norm(content)
    return [(t, u) for n, (t, u) in PROJECT_LINKS.items() if n in text]

@app.get("/")
def home():
    return {
        "message": "DevVibe AI Chatbot Backend is running!"
    }


# Words that usually do not help identify a document's topic.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "from", "give", "i", "in", "is", "it",
    "me", "of", "on", "or", "our", "please", "should", "tell",
    "that", "the", "their", "them", "there", "this", "to", "us",
    "we", "what", "when", "where", "which", "who", "why", "with",
    "would", "you", "your", "devvibe", "offer", "offers", "provide",
    "provides", "company", "about", "details",
}

SERVICE_NAMES = {
    "Software Development": (
        "software development", "custom software"
    ),
    "Website Development": (
        "website development", "web development"
    ),
    "Application Development": (
        "application development", "app development"
    ),
    "AI Development": (
        "ai development", "artificial intelligence", "machine learning"
    ),
    "IT Consultation": (
        "it consultation", "consulting", "consultation"
    ),
    "Quality Assurance": (
        "quality assurance", "software testing"
    ),
    "Robotics/IoT/IIoT Solutions": (
        "robotics", "iot", "iiot"
    ),
    "Design, CGI & VFX": (
        "design cgi", "cgi", "vfx"
    ),
}

LEGAL_PAGE_TERMS = (
    "terms and conditions", "privacy policy"
)

# Portfolio listing pages. These hold one summary per project and are the
# best source for a broad case-study question, but plain vector search
# rarely ranks them high because the words "case study" do not appear in
# the project descriptions.
CASE_STUDY_PAGE_IDS = ("P031", "P032", "P033", "P034", "P035")

# Team and About pages name the leadership. Asking "who is the CTO" does
# not surface these by embedding alone, because the marketing pages that
# mention "CTO-as-a-Service" outrank them.
LEADERSHIP_PAGE_IDS = ("P072", "P088", "P092")

LEADERSHIP_CONTENT_TERMS = (
    "chief technology officer", "chief executive officer",
    "chief operating officer", "co-founder", "founder",
)


def _fetch_documents_by_page_id(
    page_ids: tuple[str, ...],
    fallback_title: str,
) -> list[dict]:
    try:
        result = (
            supabase.table("devvibe_documents")
            .select("page_id, title, content")
            .in_("page_id", list(page_ids))
            .execute()
        )
    except Exception as error:
        print(f"Document fetch failed for {page_ids}: {error}")
        return []

    return [
        {**row, "title": str(row.get("title") or fallback_title)}
        for row in (result.data or [])
        if str(row.get("content") or "").strip()
    ]


def fetch_case_study_documents() -> list[dict]:
    return _fetch_documents_by_page_id(CASE_STUDY_PAGE_IDS, "Portfolio")


def fetch_leadership_documents() -> list[dict]:
    return _fetch_documents_by_page_id(LEADERSHIP_PAGE_IDS, "Team")


def tokenize(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9+#]+", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


def get_query_intent(question: str) -> str:
    text = question.lower()

    if any(term in text for term in (
        "social media", "social accounts", "social links",
        "linkedin", "instagram", "facebook", "follow you",
    )):
        return "social"

    if any(term in text for term in (
        "contact details", "contact information", "email address",
        "phone number", "whatsapp", "how can i contact",
        "where are you located", "your address",
    )):
        return "contact"

    if any(term in text for term in (
        "what services", "which services", "list your services",
        "services do you offer", "what do you offer",
        "what can devvibe do", "your offerings",
    )):
        return "service_list"

    if any(term in text for term in (
        "case study", "case studies", "your projects", "projects",
        "portfolio", "your work", "past work",
    )):
        return "case_study"

    if any(term in text for term in (
        "terms and conditions", "privacy policy",
    )):
        return "legal"

    # Questions about who runs the company name an executive role, e.g.
    # "who is the CTO". Keep this separate from the "CTO-as-a-Service"
    # offering, and from generic talk about founders.
    is_offering = any(
        term in text for term in ("as a service", "as-a-service")
    )
    role_titles = (
        "cto", "ceo", "coo", "chief technology officer",
        "chief executive officer", "chief operating officer",
    )
    asks_who = any(term in text for term in (
        "who is", "who's", "who are", "name of", "leadership",
        "management team", "your team", "the team behind",
    ))

    if not is_offering:
        if any(role in text for role in role_titles):
            return "leadership"

        if asks_who and any(
            role in text for role in ("founder", "co-founder")
        ):
            return "leadership"

    matched_services = [
        name for name, aliases in SERVICE_NAMES.items()
        if any(alias in text for alias in aliases)
    ]

    if matched_services:
        return "service_detail"

    return "general"


# Short messages that usually refer back to something said earlier.
FOLLOW_UP_PRONOUNS = {
    "it", "its", "this", "that", "these", "those", "them",
    "they", "he", "she", "there", "one", "ones",
}

FOLLOW_UP_PHRASES = (
    "link", "read more", "learn more","tell me more", "more about", 
    "more detail", "more details", "go on", "the first", "the second", 
    "the third", "the last", "the same", "that one", "this one", 
    "those ones", "what about", "and the", "how about", "which of", 
    "of them", "of those", "another", "also",
)


def build_retrieval_query(question: str, history: list[ChatMessage]) -> str:
    """Expand short follow-up questions with the previous user request.

    The embedding model only sees the current message, so a follow-up such
    as "tell me more about that" would otherwise retrieve unrelated chunks.
    """

    previous_user_messages = [
        message.content.strip()
        for message in history
        if message.role == "user" and message.content.strip()
    ]

    if not previous_user_messages:
        return question

    text = question.lower().strip()
    words = set(re.findall(r"[a-z0-9+#.-]+", text))

    is_short = len(question.split()) <= 12
    refers_back = (
        bool(words & FOLLOW_UP_PRONOUNS)
        or any(phrase in text for phrase in FOLLOW_UP_PHRASES)
    )
    is_bare_reply = text in {
        "yes", "no", "sure", "ok", "okay", "please", "go ahead",
        "yes please", "no thanks", "more", "and", "why", "how",
    }

    if not (is_short and (refers_back or is_bare_reply)):
        return question

    return f"{previous_user_messages[-1]} {question}"


def recent_conversation(
    history: list[ChatMessage],
    limit: int = 8,
) -> list[dict]:
    """Return the most recent well-formed turns for the chat model."""

    turns = []

    for message in history[-limit:]:
        role = message.role.strip().lower()
        content = message.content.strip()

        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content})

    return turns


def score_document(question: str, document: dict) -> int:
    title = str(document.get("title") or "")
    content = str(document.get("content") or "")
    title_lower = title.lower()
    content_lower = content.lower()
    question_tokens = tokenize(question)

    score = 4 * len(question_tokens & tokenize(title))
    score += len(question_tokens & tokenize(content))

    intent = get_query_intent(question)

    # Avoid legal pages unless the user is asking about legal policies.
    if intent != "legal" and any(
        term in title_lower for term in LEGAL_PAGE_TERMS
    ):
        return -100

    if intent == "service_list":
        # Prefer chunks that actually list multiple DevVibe services.
        service_mentions = sum(
            1 for aliases in SERVICE_NAMES.values()
            if any(alias in content_lower for alias in aliases)
        )
        score += service_mentions * 5

        if "services" in title_lower or "service" in title_lower:
            score += 8

    elif intent == "service_detail":
        # Prioritize the exact service pages named in the question.
        for name, aliases in SERVICE_NAMES.items():
            if any(alias in question.lower() for alias in aliases):
                if name.lower() in title_lower:
                    score += 25
                elif any(alias in title_lower for alias in aliases):
                    score += 20
                elif any(alias in content_lower for alias in aliases):
                    score += 3

    elif intent == "contact":
        if any(term in title_lower for term in (
            "contact", "get in touch", "contact us"
        )):
            score += 30

        if any(term in content_lower for term in (
            "connect@devvibe.com", "our locations", "follow us on"
        )):
            score += 12

        if "whatsapp" in content_lower:
            score += 5

    elif intent == "case_study":
        if any(term in title_lower for term in (
            "portfolio", "case study", "case studies", "projects"
        )):
            score += 20

    elif intent == "leadership":
        if any(term in content_lower for term in LEADERSHIP_CONTENT_TERMS):
            score += 25

        if title_lower.strip() in ("team", "about us"):
            score += 10

    return score


def retrieve_context(question: str) -> list[dict]:
    if get_query_intent(question) == "social":
        social = (
            supabase.table("devvibe_documents")
            .select("title,content")
            .eq("page_id", "SOCIAL_LINKS")
            .execute()
            .data
        )
        if social:
            return social
        
    query_embedding = list(
        embedding_model.embed([question])
    )[0].tolist()

    # Retrieve a wider pool so relevant pages have a better chance
    # of reaching the reranking stage.
    result = supabase.rpc(
        "match_documents_fastembed",
        {
            "query_embedding": query_embedding,
            "match_count": 30,
        },
    ).execute()

    candidates = result.data or []

    intent = get_query_intent(question)

    # Case-study questions need the portfolio pages, and leadership
    # questions need the Team/About pages. Both rank poorly on embeddings
    # alone because the wording differs, so add them explicitly.
    if intent == "case_study":
        candidates = candidates + fetch_case_study_documents()
    elif intent == "leadership":
        candidates = candidates + fetch_leadership_documents()

    unique_documents = []
    seen_content = set()

    for document in candidates:
        title = str(document.get("title") or "Untitled page").strip()
        content = str(document.get("content") or "").strip()

        if not content:
            continue

        normalized_content = re.sub(
            r"\s+", " ", content
        ).lower()

        if normalized_content in seen_content:
            continue

        seen_content.add(normalized_content)
        unique_documents.append({
            **document,
            "title": title,
            "content": content,
        })

    ranked_documents = sorted(
        unique_documents,
        key=lambda document: score_document(question, document),
        reverse=True,
    )

    # Exclude documents ranked as irrelevant, and don't force six
    # chunks into the prompt when fewer have a positive score.
    selected_documents = [
        document for document in ranked_documents
        if score_document(question, document) > 0
    ][:6]

    print(f"\nQuestion: {question}")
    print(f"Retrieved candidates: {len(candidates)}")
    print(f"Unique candidates: {len(unique_documents)}")
    print(f"Selected chunks: {len(selected_documents)}")

    for index, document in enumerate(selected_documents, start=1):
        print(f"\n--- Selected chunk {index} ---")
        print(f"Title: {document['title']}")
        print(f"Score: {score_document(question, document)}")
        print(f"Content: {document['content'][:700]}")

    return selected_documents


def generate_answer(messages: list[dict]) -> str | None:
    last_rate_limit_error = None

    for model in [MODEL_NAME, *FALLBACK_MODELS]:
        try:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
        except RateLimitError as error:
            print(f"Rate limit reached for {model}: {error}")
            last_rate_limit_error = error
            continue
        except APIStatusError as error:
            if error.status_code in (400, 404, 429):
                print(f"Model {model} unavailable ({error.status_code}): {error}")
                if error.status_code == 429:
                    last_rate_limit_error = error
                continue
            raise

        answer = completion.choices[0].message.content
        if answer:
            return answer

    if last_rate_limit_error is not None:
        raise last_rate_limit_error

    return None


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        retrieval_query = build_retrieval_query(question, request.history)
        documents = retrieve_context(retrieval_query)

        wants_link = any(t in question.lower() for t in (
            "link", "url", "read more", "learn more",
            "full case study", "share it", "send it",
        ))
        
        if documents:
            context_parts = []

            for document in documents:
                title = document["title"]
                content = document["content"]
                project_links = find_project_links(content)
                if wants_link and project_links:
                    content += "\n\nProject page link(s): " + "; ".join(
                        f"{t} - {u}" for t, u in project_links
                    )

                context_parts.append(
                    f"WEBSITE SECTION: {title}\n"
                    f"SECTION CONTENT:\n{content}"
                )

            context = "\n\n---\n\n".join(context_parts)
        else:
            context = (
                "No relevant website chunks were returned. "
                "The original website may still contain the answer."
            )

        system_prompt = """
You are the official DevVibe website assistant talking to a
prospective client in a live chat.

SOURCE AND ACCURACY:
- The supplied DevVibe website sections are the source of truth.
- Do not invent services, features, prices, clients, names, or
  company claims.
- Treat each section as a separate source. Do not merge or blend
  unrelated sections into one description.
- Use only the sections that actually support the answer, and prefer
  the section that matches what was asked.
- If the sections do not establish an answer, say plainly:
  "I couldn't find that information on the DevVibe website."

CONVERSATION CONTEXT:
- You are mid-conversation. Read the earlier messages and build on
  them instead of starting over on every reply.
- Resolve follow-ups ("tell me more", "the second one", "what about
  that?") against what was just discussed, and answer only what is
  now being asked.
- Do not repeat the previous answer or dump unrelated information
  that was not requested.
- If a follow-up is genuinely ambiguous, ask one short clarifying
  question instead of guessing.

SERVICE QUESTIONS:
- DevVibe offers distinct services: AI Development, Software
  Development, Application Development, Website Development,
  Design/CGI & VFX, Quality Assurance, IT Consultation, and
  Robotics/IoT/IIoT Solutions.
- For a broad "what do you offer?" question, list each service on
  its own line with its own one-sentence explanation, using the
  website's wording. Keep each service separate; never describe one
  service using details that belong to another.
- For a specific service, explain just that service in a short
  paragraph or a few bullets. Do not return the whole catalogue
  unless it was requested.
- Do not present engagement models, packages, or case studies as
  services unless the website explicitly calls them services.

TEAM AND LEADERSHIP:
- When asked who holds a role (for example the CTO, CEO, or COO),
  name the person and their exact title from the team/about content.
- DevVibe's leadership includes Faiz Jillani (Chief Executive Officer
  & Founder), Muhammad Shazil (Chief Technology Officer & Co-Founder),
  and Ajmal Jillani (Chief Operating Officer). Only state these when
  the supplied sections support it; never guess or invent other people.
- Do not confuse a person with the "CTO-as-a-Service" offering.

CASE STUDIES AND PORTFOLIO:
- When asked about case studies or projects, briefly explain each one
  in plain prose: the client or product and what we built or did.
- Mention the problem, technologies, or results ONLY if the supplied
  sections state them. Never infer or invent challenges, outcomes,
  benefits, or numbers.
- If a project's section is short, give a short description and offer
  to tell them more about it.
- Never reply with only a list of links or project titles.
- Do not mix details from one project into another, and do not attach
  general tools or technologies to projects unless the section says
  they were used there.
- Do not add portfolio items that were not asked about.
- When a "Project page link" is supplied, finish with one short line
  such as "You can read the full case study here: [Project name](url)".
  Use only links that were supplied. Never invent or guess a URL.
- If the user asks for a link, give the supplied link for the project
  being discussed.

RESPONSE STYLE:
- Be brief. Most answers are 2-3 sentences, under 60 words.
- Sound like a person in a live chat: natural, warm, direct. Speak
  as DevVibe ("we", "our").
- Give only the most meaningful facts. Skip process steps, tool
  lists, and minor details unless the user asks for them.
- Never use headings or sections like "What we built" or "Key
  outcomes".
- Do not include links, "read more" lines, or URLs unless the user
  asks for a link. When they do, give the supplied link with one
  short sentence about what the page contains.
- Exception: when asked about social media, give each supplied
  account as a Markdown link, for example [LinkedIn](url).
- Always format lists of services or projects as a Markdown bullet
  list. Start each line with "- " and put the name in bold, like:
  "- **Software Development** – one short sentence."
- For services, list all eight. For projects, list at most 5.
  Then offer to explain any of them.
- If the user asks for names only, give only names.
- Do not repeat earlier messages. End with at most one short offer
  to help further, only if useful.
- Use bold sparingly. No tables, no HTML.

EXAMPLES OF THE RIGHT LENGTH:
Q: Tell me about the Nanoveu project
A: We built a modern corporate website for Nanoveu, an ASX-listed
nanotechnology company. It's fast, scalable, and integrated with
HubSpot, and it doubled user engagement while strengthening their
investor credibility. Want more detail on any part?

Q: Tell me about your AI Development service
A: We take AI ideas from concept to a working product: building
custom models, setting up the infrastructure to train and deploy
them, and supporting them afterwards. Want to hear about a
specific part?
"""

        conversation = recent_conversation(request.history)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *conversation,
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Relevant DevVibe website sections:\n{context}"
                ),
            },
        ]

        try:
            answer = generate_answer(messages)
        except RateLimitError:
            return {"response": BUSY_MESSAGE}
        except APIStatusError as error:
            if error.status_code == 429:
                return {"response": BUSY_MESSAGE}
            raise
        except (APIConnectionError, APITimeoutError):
            return {"response": CONNECTION_MESSAGE}

        if not answer:
            answer = (
                "I couldn't generate a response. Please try again."
            )

        return {"response": answer.strip()}

    except HTTPException:
        raise

    except Exception as error:
        print(f"Chat endpoint error: {error}")

        raise HTTPException(
            status_code=500,
            detail=(
                "The chatbot encountered an error while processing "
                "your question."
            ),
        )