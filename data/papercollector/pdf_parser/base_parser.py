from pathlib import Path
from typing import List, Tuple

from rich.progress import track
import os
import re

from papercollector.db import BaseDB


class BasePdfParser:
    """Base PDF parser class."""

    def __init__(self, directory: Path, db: BaseDB) -> None:
        """Create a BasePdfParser.

        Args:
            directory (Path): The directory where the pdfs are stored.
            db (BaseDB): The database where to save data.
        """
        self.rootdir = directory
        self.db = db
        self.MIN_BLOCK_LEN_WORDS = 20
        self.parsed_files = []
        self.unparsable_files = []

    def delete_parsed(self) -> None:
        """Delete the parsed pdfs."""
        for f in track(self.parsed_files, description="Deleting parsed pdfs"):
            os.remove(f)
        self.parsed_files = []

    def delete_unparsable(self) -> None:
        """Delete the unparsable pdfs."""
        for f in track(self.unparsable_files, description="Deleting unparsable pdfs"):
            os.remove(f)
        self.unparsable_files = []

    def _is_section_title(self, content: str, word_list: List[str]) -> bool:
        """Decide if a section is a title.

        Args:
            content (str): The contect of the section.
            word_list (List[str]): The list of words in the content (content.split()).

        Returns:
            bool: True if the section is a title.
        """
        if content.replace(" ", "").lower() in ["abstract", "references"]:
            return True
        is_short = len(word_list) < self.MIN_BLOCK_LEN_WORDS
        normal = re.compile("^\d+\.?\d*( |\n)+\w.*$")
        roman = re.compile(
            "^(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))+\.?( |\n)+\w.*$"
        )
        return is_short and (re.match(normal, content) or re.match(roman, content))

    def start(self) -> Tuple[int, int]:
        """Start the parsing

        Returns:
            Tuple[int, int]: A tuple where the first element indicates how many pdfs
            were parsed and the second how many failures.
        """
        raise NotImplementedError("Must be implemented by a Parser object")
