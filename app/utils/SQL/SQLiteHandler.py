import sqlite3
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from typing import Literal, Dict, Any, Union





class SQLiteHandler:
    """
    A handler class for SQLite database operations.
    Attributes:
        db_name (str): The name of the SQLite database file.
        can_read (bool): Flag indicating if read operations are allowed. Default is True.
        can_write (bool): Flag indicating if write operations are allowed. Default is True.
        conn (sqlite3.Connection): The SQLite database connection object.
        cursor (sqlite3.Cursor): The SQLite database cursor object.
    Methods:
        close_connection():
            Closes the database connection.
        get_complete_Dataframe(table_name):
            Retrieves the entire table as a pandas DataFrame.
            Args:
                table_name (str): The name of the table to retrieve.
            Returns:
                pandas.DataFrame: The table data as a DataFrame.
        execute_query(query):
            Executes a given SQL query.
            Args:
                query (str): The SQL query to execute.
        ensure_database_exists():
            Checks if the database file exists, and creates it if it doesn't.
    """
    def __init__(self, db_name, can_read=True, can_write=True):
        self.db_name = db_name
        self.can_read = can_read
        self.can_write = can_write
        self.ensure_database_exists()
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Connected to database {self.db_name}")


    def fetch_distinct_values(self, table: str, column: str):
        sql = f"SELECT DISTINCT {column} FROM {table};"
        return {row[0] for row in self.conn.execute(sql)}

    def ensure_database_exists(self):
        directory = os.path.dirname(self.db_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist
            logging.info(f"Created directory {directory} for database {self.db_name}")
        if not os.path.exists(self.db_name):
            open(self.db_name, 'w').close()
            logging.info(f"Database {self.db_name} did not exist and was created.")

    def close_connection(self):
        self.conn.close()
        logging.debug1(f"Closed connection to database {self.db_name}")

    def get_complete_Dataframe(self, table_name, first_n_rows=None, every_nth_row=None):
        if first_n_rows is not None:
            query = f'SELECT * FROM "{table_name}" LIMIT {first_n_rows};'

        elif every_nth_row is not None:
            query = f'''
                SELECT *
                FROM (
                    SELECT *, ROW_NUMBER() OVER () AS rn
                    FROM "{table_name}"
                ) sub
                WHERE MOD(rn - 1, {every_nth_row}) = 0;
            '''

        else:
            query = f'SELECT * FROM "{table_name}";'

        dataframe = pd.read_sql_query(query, self.conn)
        logging.info(f"Retrieved data from table {table_name}")
        return dataframe

    
    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            self.conn.commit()
            logging.info(f"Executed query: {query}")
        except Exception as e:
            logging.error(f"Failed to execute query: {query} with error: {e}")

    def store_table(self, table_name, dataframe, method: Literal["replace", "append"]):
        """
        Stores the given dataframe into the specified table in the database.

        Parameters:
        table_name (str): The name of the table where the dataframe will be stored.
        dataframe (pandas.DataFrame): The dataframe to be stored in the table.
        method (str): The method to use for storing the dataframe. Currently, only 'replace' is supported.

        Raises:
        PermissionError: If the instance is write-protected.
        ValueError: If an invalid method is provided.

        Logs:
        Info: When the table is successfully replaced with new data.
        Error: When an invalid method is provided.
        """
        if not self.can_write:
            raise PermissionError("This instance is write-protected.")
        if method == 'replace':
            dataframe.to_sql(table_name, self.conn, if_exists='replace', index=False)
            logging.info(f"Replaced table {table_name} with new data")
        elif method == 'append':
            dataframe.to_sql(table_name, self.conn, if_exists='append', index=False)
            logging.info(f"Appended data to table {table_name}")
        else:
            logging.error(f"Invalid method: {method}. Please use 'replace' or 'append'.")

    def get_table_names(self):
        """
        Retrieves the names of all tables in the database.

        Returns:
        list: A list of table names in the database.
        """
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        self.cursor.execute(query)
        tables = [row[0] for row in self.cursor.fetchall()]
        logging.debug1(f"Retrieved table names: {tables}")
        return tables
    
    def delete_table(self, table_name):
        """
        Deletes the specified table from the database.

        Parameters:
        table_name (str): The name of the table to be deleted.
        """
        query = f"DROP TABLE IF EXISTS {table_name};"
        self.cursor.execute(query)
        self.conn.commit()
        logging.info(f"Deleted table {table_name}")

    @staticmethod
    def get_one(
        db_path: str,
        table: str,
        key_col: str,
        key_val: Union[str, int],
    ) -> Dict[str, Any]:
        """
        Return a single row from `table` in `db_path` where `key_col` == `key_val`.

        Raises
        ------
        ValueError  : if 0 or >1 rows are found
        sqlite3.Error : on connection / query issues
        """
        query = f"SELECT * FROM {table} WHERE {key_col} = ?"

        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(key_val,))

        if len(df) == 0:
            raise ValueError(f"No row found for {key_col} = {key_val!r}")
        if len(df) > 1:
            raise ValueError(f"Expected one row, found {len(df)} for {key_val!r}")

        return df.iloc[0].to_dict()
