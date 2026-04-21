from sqlalchemy import text
from ..connection import get_async_connection


async def fetch_data():
    async with get_async_connection() as connection:
        statement = text("""SELECT * FROM da_sampling.real""")

        data = await connection.execute(statement=statement)

    return data.all()
