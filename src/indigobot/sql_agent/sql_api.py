"""
FastAPI application providing a REST API for the Chinook music database.

This module implements CRUD operations for artists, genres, albums, and tracks.
It uses SQLAlchemy for database operations and Pydantic for data validation.
"""

import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from indigobot.config import RAG_DIR
from .init_db import (
    Album,
    Artist,
    Base,
    Genre,
    SessionLocal,
    Track,
    engine,
    init_sql_db,
)

# Ensure database directory exists
os.makedirs(RAG_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Initialize database with sample data
init_sql_db()


# Define Pydantic Models
class ArtistCreate(BaseModel):
    Name: str


class ArtistResponse(BaseModel):
    ArtistId: int
    Name: str

    class Config:
        from_attributes = True


class GenreCreate(BaseModel):
    Name: str


class GenreResponse(BaseModel):
    GenreId: int
    Name: str

    class Config:
        from_attributes = True


class AlbumCreate(BaseModel):
    Title: str
    ArtistId: int


class AlbumResponse(BaseModel):
    AlbumId: int
    Title: str
    ArtistId: int

    class Config:
        from_attributes = True


class TrackCreate(BaseModel):
    Name: str
    AlbumId: int
    GenreId: int
    Composer: str | None = None
    Milliseconds: int
    Bytes: int
    UnitPrice: str


class TrackResponse(BaseModel):
    TrackId: int
    Name: str
    AlbumId: int
    GenreId: int
    Composer: str | None
    Milliseconds: int
    Bytes: int
    UnitPrice: str

    class Config:
        from_attributes = True


# Create database tables
Base.metadata.create_all(bind=engine)


# Dependency to get database session
def get_api_db():
    """
    Dependency that provides a database session.

    Yields:
        SessionLocal: A database session object.

    Ensures that the database session is closed after the request is processed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# API Routes
@app.get("/")
def read_root():
    """
    Get API welcome message and list of available endpoints.

    Returns:
        dict: A dictionary containing:
            - message: Welcome message
            - endpoints: Dictionary of available API endpoints and their descriptions
    """
    return {
        "message": "Welcome to the Chinook API",
        "endpoints": {
            "GET /": "This help message",
            "GET /artists": "List all artists",
            "GET /artists/{id}": "Get specific artist",
            "POST /artists": "Create new artist",
            "GET /genres": "List all genres",
            "GET /genres/{id}": "Get specific genre",
            "POST /genres": "Create new genre",
            "GET /albums": "List all albums",
            "GET /albums/{id}": "Get specific album",
            "POST /albums": "Create new album",
            "GET /tracks": "List all tracks",
            "GET /tracks/{id}": "Get specific track",
            "POST /tracks": "Create new track",
        },
    }


@app.post("/artists/", response_model=ArtistResponse)
def create_artist(artist: ArtistCreate, db: SessionLocal = Depends(get_api_db)):
    """
    Create a new artist in the database.

    Args:
        artist (ArtistCreate): An object containing the name of the artist.
        db (SessionLocal): Database session dependency.

    Returns:
        ArtistResponse: The created artist details.
    """
    db_artist = Artist(**artist.dict())
    db.add(db_artist)
    db.commit()
    db.refresh(db_artist)
    return db_artist


@app.get("/artists/{artist_id}", response_model=ArtistResponse)
def read_artist(artist_id: int, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a specific artist by their ID.

    Args:
        artist_id (int): The unique identifier of the artist
        db (SessionLocal): Database session dependency

    Returns:
        ArtistResponse: The artist details

    Raises:
        HTTPException: 404 if artist not found
    """
    artist = db.query(Artist).filter(Artist.ArtistId == artist_id).first()
    if artist is None:
        raise HTTPException(status_code=404, detail="Artist not found")
    return artist


@app.get("/artists/", response_model=list[ArtistResponse])
def read_artists(
    skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_api_db)
):
    """
    Retrieve a list of artists with pagination.

    Args:
        skip (int): Number of records to skip (default: 0)
        limit (int): Maximum number of records to return (default: 10)
        db (SessionLocal): Database session dependency

    Returns:
        list[ArtistResponse]: List of artists
    """
    artists = db.query(Artist).offset(skip).limit(limit).all()
    return artists


@app.post("/genres/", response_model=GenreResponse)
def create_genre(genre: GenreCreate, db: SessionLocal = Depends(get_api_db)):
    """
    Create a new genre in the database.

    Args:
        genre (GenreCreate): An object containing the name of the genre.
        db (SessionLocal): Database session dependency.

    Returns:
        GenreResponse: The created genre details.
    """
    db_genre = Genre(**genre.dict())
    db.add(db_genre)
    db.commit()
    db.refresh(db_genre)
    return db_genre


@app.get("/genres/{genre_id}", response_model=GenreResponse)
def read_genre(genre_id: int, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a specific genre by its ID.

    Args:
        genre_id (int): The unique identifier of the genre
        db (SessionLocal): Database session dependency

    Returns:
        GenreResponse: The genre details

    Raises:
        HTTPException: 404 if genre not found
    """
    genre = db.query(Genre).filter(Genre.GenreId == genre_id).first()
    if genre is None:
        raise HTTPException(status_code=404, detail="Genre not found")
    return genre


@app.get("/genres/", response_model=list[GenreResponse])
def read_genres(skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a list of genres with pagination.

    Args:
        skip (int): Number of records to skip (default: 0)
        limit (int): Maximum number of records to return (default: 10)
        db (SessionLocal): Database session dependency

    Returns:
        list[GenreResponse]: List of genres
    """
    genres = db.query(Genre).offset(skip).limit(limit).all()
    return genres


@app.post("/tracks/", response_model=TrackResponse)
def create_track(track: TrackCreate, db: SessionLocal = Depends(get_api_db)):
    """
    Create a new track in the database.

    Args:
        track (TrackCreate): An object containing the track details including name,
            album ID, genre ID, composer, duration, size, and price.
        db (SessionLocal): Database session dependency.

    Returns:
        TrackResponse: The created track details.
    """
    db_track = Track(**track.dict())
    db.add(db_track)
    db.commit()
    db.refresh(db_track)
    return db_track


@app.get("/tracks/{track_id}", response_model=TrackResponse)
def read_track(track_id: int, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a specific track by its ID.

    Args:
        track_id (int): The unique identifier of the track
        db (SessionLocal): Database session dependency

    Returns:
        TrackResponse: The track details

    Raises:
        HTTPException: 404 if track not found
    """
    track = db.query(Track).filter(Track.TrackId == track_id).first()
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@app.get("/tracks/", response_model=list[TrackResponse])
def read_tracks(skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a list of tracks with pagination.

    Args:
        skip (int): Number of records to skip (default: 0)
        limit (int): Maximum number of records to return (default: 10)
        db (SessionLocal): Database session dependency

    Returns:
        list[TrackResponse]: List of tracks
    """
    tracks = db.query(Track).offset(skip).limit(limit).all()
    return tracks


@app.post("/albums/", response_model=AlbumResponse)
def create_album(album: AlbumCreate, db: SessionLocal = Depends(get_api_db)):
    """
    Create a new album in the database.

    Args:
        album (AlbumCreate): An object containing the album title and artist ID.
        db (SessionLocal): Database session dependency.

    Returns:
        AlbumResponse: The created album details.
    """
    db_album = Album(**album.dict())
    db.add(db_album)
    db.commit()
    db.refresh(db_album)
    return db_album


@app.get("/albums/{album_id}", response_model=AlbumResponse)
def read_album(album_id: int, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a specific album by its ID.

    Args:
        album_id (int): The unique identifier of the album
        db (SessionLocal): Database session dependency

    Returns:
        AlbumResponse: The album details

    Raises:
        HTTPException: 404 if album not found
    """
    album = db.query(Album).filter(Album.AlbumId == album_id).first()
    if album is None:
        raise HTTPException(status_code=404, detail="Album not found")
    return album


@app.get("/albums/", response_model=list[AlbumResponse])
def read_albums(skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_api_db)):
    """
    Retrieve a list of albums with pagination.

    Args:
        skip (int): Number of records to skip (default: 0)
        limit (int): Maximum number of records to return (default: 10)
        db (SessionLocal): Database session dependency

    Returns:
        list[AlbumResponse]: List of albums
    """
    albums = db.query(Album).offset(skip).limit(limit).all()
    return albums


if __name__ == "__main__":
    uvicorn.run("sql_agent.sql_api:app", host="0.0.0.0", port=8000, reload=True)
