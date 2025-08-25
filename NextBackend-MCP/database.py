from sqlalchemy import create_engine, Column, Integer, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# -----------------------------
# SQLite database
# -----------------------------
SQLALCHEMY_DATABASE_URL = "sqlite:///./content.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# -----------------------------
# ORM Model
# -----------------------------
class Content(Base):
    __tablename__ = "contents"

    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, index=True)        # "youtube" / "web"
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    url = Column(String, unique=True, nullable=False, index=True)
    thumbnail = Column(String, nullable=True)
    embedding = Column(JSON, nullable=True)      # store embedding as JSON list of floats


# -----------------------------
# Dependency
# -----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# Create tables
# -----------------------------
Base.metadata.create_all(bind=engine)
