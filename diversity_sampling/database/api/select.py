from sqlalchemy import text
from ..connection import get_async_connection


async def get_augment_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM core_sets.augment_set""")

        data = await connection.execute(statement=statement)

    return data.all()


async def get_retain_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM core_sets.retain_set""")

        data = await connection.execute(statement=statement)

    return data.all()


async def get_high_quality_synthetic_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM augment_sets.high_quality""")

        data = await connection.execute(statement=statement)

    return data.all()


async def get_test_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM downstream.test_set""")

        data = await connection.execute(statement=statement)

    return data.all()
