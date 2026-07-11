import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.knowledge.knowledge import Knowledge
from agno.models.groq import Groq
from agno.vectordb.pgvector import PgVector

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------

load_dotenv()

load_dotenv()

groq_key = os.getenv("GROQ")

if not groq_key:
    raise ValueError("GROQ API key not found in .env")

os.environ["GROQ_API_KEY"] = groq_key

# -------------------------------------------------
# Database URL
# -------------------------------------------------

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# -------------------------------------------------
# Contents Database
# -------------------------------------------------

contents_db = PostgresDb(
    db_url=db_url,
    knowledge_table="knowledge_contents",
)

# -------------------------------------------------
# Vector Database
# -------------------------------------------------

vector_db = PgVector(
    table_name="recipes",
    db_url=db_url,
)

# -------------------------------------------------
# Create Knowledge Base
# -------------------------------------------------

knowledge = Knowledge(
    name="Recipe Knowledge",
    description="Recipe PDF Knowledge Base",
    contents_db=contents_db,
    vector_db=vector_db,
)

# -------------------------------------------------
# Load PDF (Run only once)
# -------------------------------------------------

knowledge.insert(
    name="Recipe Book",
    url="https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf",
)

# -------------------------------------------------
# Create Agent
# -------------------------------------------------

agent = Agent(
    model=Groq(
        id="llama-3.1-8b-instant",
    ),
    db=contents_db,
    knowledge=knowledge,
    search_knowledge=True,
    add_history_to_context=True,
    markdown=True,
)

# -------------------------------------------------
# Chat Loop
# -------------------------------------------------

while True:

    query = input("\nYou : ")

    if query.lower() in ["exit", "quit"]:
        break

    agent.print_response(query)