import sqlite3
from pathlib import Path

import pandas as pd


class BaseDB:
    """
    A base class for the papers DB. By default this class only includes
    id, title, abstract, introduction and conclusion.
    If you want to extend the database extend the class and overwrite the methods.
    """

    def __init__(self, directory: Path, name: str):
        """Create a BaseDB object, connect to the backend, and create papers table.

        Args:
            directory (Path): The directory where the sqlite file is stored.
            name (str): The name of the sqlite file.
        """
        self.path = directory / (name + ".db")
        self.connect()
        self._create_papers_table()

    def _create_papers_table(self) -> None:
        """Create the papers table if it does not exists"""
        if not self.table_exists("papers"):
            self.cursor.execute(
                """
                    CREATE TABLE papers (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        abstract TEXT NOT NULL,
                        introduction TEXT NOT NULL,
                        conclusion TEXT NOT NULL
                    );
                """
            )

    def insert_paper(
        self,
        id: str,
        title: str,
        abstract: str,
        introduction: str,
        conclusion: str,
    ) -> None:
        """Insert a new paper into the database.

        Args:
            id (str): The id of the paper.
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.
            introduction (str): The introduction of the paper.
            conclusion (str): The conclusion of the paper.
        """
        query: str = """
            INSERT OR IGNORE INTO papers
            (id, title, abstract, introduction, conclusion)
            VALUES (?, ?, ?, ?, ?);
        """
        self.cursor.execute(
            query,
            (id, title, abstract, introduction, conclusion),
        )

    def connect(self) -> None:
        """Connect to the sqlite backend."""
        self.connection_open = True
        self.connection: sqlite3.Connection = sqlite3.connect(self.path)
        self.cursor: sqlite3.Cursor = self.connection.cursor()

    def table_exists(self, table: str) -> bool:
        """Check if a table exits.

        Args:
            table (str): The table name.

        Returns:
            bool: True if the table exits.
        """
        if not self.connection_open:
            self.connect()

        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?;",
            (table,),
        )
        return self.cursor.fetchone() is not None

    def index_exists(self, name: str) -> bool:
        """Check if an index exits.

        Args:
            name (str): The index name.

        Returns:
            bool: True if the table exits.
        """
        if not self.connection_open:
            self.connect()

        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?;",
            (name,),
        )
        return self.cursor.fetchone() is not None

    def is_table_empty(self, table: str) -> bool:
        """Check if a table is empty.

        Args:
            table (str): The table name.

        Returns:
            bool: True if the table is empty.
        """
        if not self.connection_open:
            self.connect()
        self.cursor.execute("SELECT count(*) FROM (SELECT 0 FROM ? LIMIT 1);", (table))
        return self.cursor.fetchone()[0] == 0

    def commit(self) -> None:
        """Commit changes to the database."""
        if self.connection_open:
            self.connection.commit()

    def close(self) -> None:
        """Close the sqlite connection."""
        if self.connection_open:
            self.connection_open = False
            self.connection.close()

    def commit_and_close(self) -> None:
        """Commit changes to the database and close sqlite connection."""
        if self.connection_open:
            self.commit()
            self.close()

    def get_papers_df(self) -> pd.DataFrame:
        query: str = "SELECT * FROM papers ORDER BY RANDOM();"
        df = pd.read_sql_query(
            query,
            self.connection,
            dtype=str,
        )
        return df
