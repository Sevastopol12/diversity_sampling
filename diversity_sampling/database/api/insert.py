import pandas as pd
from diversity_sampling.database.connection import get_connection


def insert_table(
    table_name: str, table_data: pd.DataFrame, schema: str = "da_sampling"
):
    with get_connection() as connection:
        table_data.to_sql(
            con=connection.get_bind(),
            schema=schema,
            name=table_name,
            if_exists="replace",
            index=False,
        )
