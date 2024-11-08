import os

import polars as pl
from databricks import sql
from dotenv import load_dotenv

load_dotenv()


def read_from_databricks(query: str):
    """Fetch data from Databricks using the provided query.

    Parameters
    ----------
    query : str
        The query to be executed on Databricks.

    Returns
    -------
    pl.DataFrame
        The data fetched from Databricks.

    """
    connection = sql.connect(
        server_hostname=os.getenv("DATABRICKS_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_ACCESS_TOKEN"),
    )
    cursor = connection.cursor()

    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    cursor.close()
    connection.close()

    return pl.DataFrame(data, strict=True, schema=column_names)
