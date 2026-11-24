"""Build the whole assistant in one call. Used by notebooks, the test runner and (later) the web app."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.assistant import Assistant
from src.rag.generator import Generator
from src.rag.knowledge_base import load_chunks
from src.rag.models import Embedder
from src.rag.retriever import Retriever
from src.rag.rewriter import QueryRewriter
from src.support import TicketStore

GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def build_assistant(root=".", model=DEFAULT_MODEL, tickets_file="data/tickets.jsonl") -> Assistant:
    """root = the project folder (".." when called from a notebook in notebooks/)."""
    root = Path(root)
    load_dotenv(root / ".env")
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_URL, max_retries=5)
    retriever = Retriever(load_chunks(root / "knowledge_base"), embedder=Embedder("BAAI/bge-m3"))
    return Assistant(retriever=retriever,
                     generator=Generator(client, model=model),
                     rewriter=QueryRewriter(client, model=model),
                     tickets=TicketStore(root / tickets_file))
