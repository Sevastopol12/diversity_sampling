import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import DatabaseError, SQLAlchemyError
from contextlib import asynccontextmanager, contextmanager


db_url: str = os.getenv("DATABASE_URL")


async_engine = create_async_engine(url=db_url)
engine = create_engine(url=db_url)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, autoflush=False, autocommit=False
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@asynccontextmanager
async def get_async_connection():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except (DatabaseError, SQLAlchemyError) as e:
            await session.rollback()
            print(e)
            raise


@contextmanager
def get_connection():
    with SessionLocal() as session:
        try:
            yield session
            session.commit()
        except (DatabaseError, SQLAlchemyError) as e:
            session.rollback()
            print(e)
            raise
