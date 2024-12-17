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


# Define SQLAlchemy Models
class Artist(Base):
    __tablename__: str = "artists"
    ArtistId: Column = Column(Integer, primary_key=True)
    Name: Column = Column(String)


class Genre(Base):
    __tablename__: str = "genres"
    GenreId: Column = Column(Integer, primary_key=True)
    Name: Column = Column(String)


class Album(Base):
    __tablename__: str = "albums"
    AlbumId: Column = Column(Integer, primary_key=True)
    Title: Column = Column(String, nullable=False)
    ArtistId: Column = Column(Integer, nullable=False)


class Track(Base):
    __tablename__: str = "tracks"
    TrackId: Column = Column(Integer, primary_key=True, autoincrement=True)
    Name: Column = Column(String)
    AlbumId: Column = Column(Integer)
    GenreId: Column = Column(Integer)
    Composer: Column = Column(String)
    Milliseconds: Column = Column(Integer)
    Bytes: Column = Column(Integer)
    UnitPrice: Column = Column(Integer)


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
