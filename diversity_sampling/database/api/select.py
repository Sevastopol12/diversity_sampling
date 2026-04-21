from sqlalchemy import text
from ..connection import get_async_connection


async def fetch_data():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM da_sampling.real""")

        data = await connection.execute(statement=statement)

    return data.all()


async def get_augment_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM da_sampling.augment_set""")

        data = await connection.execute(statement=statement)

    return data.all()


async def get_retain_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM da_sampling.retain_set""")

        data = await connection.execute(statement=statement)

    return data.all()


async def get_prune_set():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM da_sampling.prune_set""")

        data = await connection.execute(statement=statement)

    return data.all()
