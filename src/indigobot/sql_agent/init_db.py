import os

from pydantic import SkipValidation
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from indigobot.config import GPT_DB

# Configure SQLAlchemy with Chinook database
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.abspath(GPT_DB)}?check_same_thread=False"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = SkipValidation[
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
]

Base = declarative_base()


def init_sql_db():
    """Initialize connection to the existing Chinook database"""
    db_path = GPT_DB

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    # Create a session
    session = SessionLocal()
    return session


if __name__ == "__main__":
    init_sql_db()
