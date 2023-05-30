from typing import Optional, List
from pathlib import Path

import pandas as pd
import numpy as np

from papercollector.db.base_db import BaseDB


class ArxivDB(BaseDB):
    """An arXiv database."""

    def __init__(self, directory: Path):
        """Create a BaseDB object, connect to the backend, and create papers table.

        Args:
            directory (Path): The directory where the sqlite file is stored.
        """
        super().__init__(directory, "arxiv")

    def _create_papers_table(self) -> None:
        """Create the papers table if it does not exists"""
        if not self.table_exists("papers"):
            self.cursor.execute(
                """
                    CREATE TABLE papers (
                        id TEXT PRIMARY KEY,
                        year INTEGER NOT NULL,
                        destination TEXT,
                        is_pdf_parsed INTEGER NOT NULL,
                        doi TEXT,
                        last_version TEXT NOT NULL,
                        title TEXT NOT NULL,
                        categories TEXT,
                        abstract TEXT NOT NULL,
                        introduction TEXT,
                        conclusion TEXT
                    );
                """
            )

    def complete_paper(self, id: str, introduction: str, conclusion: str) -> None:
        """Complete a previously inserted paper adding introduction and conclusion.

        Args:
            id (str): The id of the paper.
            introduction (str): The introduction.
            conclusion (str): The conclusion.
        """
        # Check destination is null or real
        query: str = """
            SELECT destination FROM papers
            WHERE id = ?;
        """
        self.cursor.execute(query, (id,))
        destination: Optional[str] = self.cursor.fetchone()[0]
        if destination is None or destination == "real":
            # Update
            introduction = introduction.encode("utf-8", "replace").decode()
            conclusion = conclusion.encode("utf-8", "replace").decode()
            query: str = """
                UPDATE papers
                SET introduction = ? , conclusion = ? , is_pdf_parsed = 1, destination = 'real'
                WHERE id = ?;
            """
            self.cursor.execute(
                query,
                (introduction, conclusion, id),
            )
        else:
            print(f"The paper {id} has {destination} as destination, skipping...")

    def count_parsed_papers(self) -> int:
        """Count the number of parsed papers in the database.

        Returns:
            int: The number of parsed papers.
        """
        query: str = """
            SELECT count(*) FROM papers
            WHERE is_pdf_parsed = 1;
        """
        self.cursor.execute(query)
        return int(self.cursor.fetchone()[0])

    def insert_paper(
        self,
        id: str,
        year: int,
        doi: Optional[str],
        last_version: str,
        title: str,
        categories: Optional[str],
        abstract: str,
    ) -> None:
        """Insert a new paper into the database.

        Args:
            id (str): The id of the paper.
            year (int): The year of the paper.
            doi (Optional[str]): The doi of the paper.
            last_version (str): The last version.
            title (str): The title of the paper.
            categories (Optional[str]): The categories of the paper.
            abstract (str): The abstract of the paper.
        """
        query: str = """
            INSERT OR IGNORE INTO papers
            (id, year, is_pdf_parsed, doi, last_version, title, categories, abstract)
            VALUES (?, ?, 0, ?, ?, ?, ?, ?);
        """
        self.cursor.execute(
            query,
            (id, year, doi, last_version, title, categories, abstract),
        )

    def papers_to_download(
        self,
        number: int,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        categories: Optional[str] = None,
        skip_unparsable: bool = True,
        skip_parsed: bool = True,
    ) -> List[str]:
        """Get the urls of papers to download which respects the filters.

        Args:
            number (int): The number of urls to generate.
            year_from (Optional[int]): The year from (inclusive).
            year_to (Optional[int]): The year to (inclusive).
            categories (Optional[str]): The categories (space separated list).
            skip_unparsable (bool): If True skip unparsable papers. Default to True.
            skip_parsed (bool): If True skip parsed papers. Default to True.

        Returns:
            List[str]: The list of urls to the papers psfs.
        """
        # Build query
        query = "SELECT id, last_version FROM papers WHERE is_pdf_parsed = 0 AND (destination IS NULL or destination = 'real')"

        if not skip_unparsable:
            query += " OR is_pdf_parsed = -1"
        if not skip_parsed:
            query += " OR is_pdf_parsed = 1"

        if categories is not None and categories != "":
            categories: List[str] = categories.split(" ")
            query += " AND ("
            for _ in range(0, len(categories) - 1):
                query += " categories LIKE ? OR"
            query += " categories LIKE ?)"

        if year_from is not None:
            query += " AND year >= ?"
        if year_to is not None:
            query += " AND year <= ?"
        query += " ORDER BY RANDOM() LIMIT ?;"

        # Build args tuple
        args = tuple()
        if categories is not None:
            args += tuple(["%" + c + "%" for c in categories])
        if year_from is not None:
            args += (year_from,)
        if year_to is not None:
            args += (year_to,)
        args += (number,)

        self.cursor.execute(query, args)

        res = []
        for item in self.cursor.fetchall():
            id, version = item[0], item[1]
            category = "arxiv"
            if "/" in id:
                splitted = id.split("/")
                category, id = splitted[0], splitted[1]

            folder = id[:4]

            res.append(
                f"gs://arxiv-dataset/arxiv/{category}/pdf/{folder}/{id}{version}.pdf"
            )

        return res

    def mark_unparsable(self, id: str):
        """Mark a paper as unparsable. Set is_pdf_parsed to -1."""
        query: str = """
            UPDATE papers
            SET is_pdf_parsed = -1
            WHERE id = ?;
        """
        self.cursor.execute(
            query,
            (id,),
        )

    def create_indexes(self) -> None:
        """Create an index in the database for the categories colum of the papers table."""
        if not self.connection_open:
            self.connect()

        if not self.index_exists("papers_categories"):
            self.cursor.execute("CREATE INDEX papers_categories ON papers(categories);")
        if not self.index_exists("papers_year"):
            self.cursor.execute("CREATE INDEX papers_year ON papers(year);")
        if not self.index_exists("papers_is_pdf_parsed"):
            self.cursor.execute(
                "CREATE INDEX papers_is_pdf_parsed ON papers(is_pdf_parsed);"
            )
        if not self.index_exists("papers_destination"):
            self.cursor.execute(
                "CREATE INDEX papers_destination ON papers(destination);"
            )

    def get_destinations(self) -> List[str]:
        """Get the not NULL destinations from the database.

        Returns:
            List[str]: The list of destinations.
        """
        query: str = """SELECT DISTINCT destination FROM papers;"""
        self.cursor.execute(query)
        destinations: List[str] = [
            x[0] for x in self.cursor.fetchall() if x[0] is not None
        ]
        return destinations

    def get_papers_df_with_destination(self, destination: str) -> pd.DataFrame:
        """Get a data frame with the papers of one destination.

        Args:
            destination (str): The destination of the papers to include in the data frame.

        Returns:
            pd.DataFrame: The data frame.
        """
        query: str = """
            SELECT id, year, title, categories FROM papers
            WHERE destination = ?
            ORDER BY RANDOM();"""
        dtypes = {
            "id": str,
            "year": np.int16,
            "title": str,
            "categories": str,
        }
        
        if destination == "real" or destination.startswith('real'):
            query = """
                SELECT id, year, title, abstract, introduction, conclusion, categories FROM papers
                WHERE destination = ?
                ORDER BY RANDOM();"""
            dtypes["introduction"] = str
            dtypes["conclusion"] = str
            dtypes["abstract"] = str

        df = pd.read_sql_query(
            query,
            self.connection,
            params = (destination,),
            dtype = dtypes,
        )
        return df

    def reserve_for_destination(self, destination: str, n: int) -> None:
        """Reserve n papers for a destination.

        Args:
            destination (str): The destination name.
            n (int): The number of papers to reserve.
        """
        query: str = """
            UPDATE papers
            SET destination = ?
            WHERE destination IS NULL
            ORDER BY RANDOM()
            LIMIT ?;
        """
        self.cursor.execute(
            query,
            (destination, n),
        )
