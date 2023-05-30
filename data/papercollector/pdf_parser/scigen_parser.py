import os
from typing import Tuple, Optional, List, Dict
import re

import fitz
from rich.progress import track

from papercollector.pdf_parser.base_parser import BasePdfParser


class ScigenPdfParser(BasePdfParser):
    """SCIgen PDFs parser class."""

    def _get_title_and_blocks(self, doc: fitz.Document) -> Tuple[str, List[str]]:
        """Get the title and the list of blocks from a document.

        Args:
            doc (fitz.Document): The document.

        Returns:
            Tuple[str, List[str]]: The tile and the list of blocks.
        """
        title = None
        blocks = []
        for page in doc:
            for block in page.get_text("blocks"):
                content = block[4]
                content = re.sub(r"-\n", "", content)
                content = re.sub(r"(?<!\r)\r(?!\r)", " ", content)
                content = re.sub(r"(?<!\n)\n(?!\n)", " ", content).strip()

                # Get title as first block of first page
                if title is None:
                    title = content
                    continue

                word_list = content.split()
                if (
                    self._is_section_title(content, word_list)
                    or len(word_list) > self.MIN_BLOCK_LEN_WORDS
                ):
                    blocks.append(content)

        return title, blocks

    def _search_sections_in_blocks(self, blocks: List[str]) -> Dict[str, Optional[str]]:
        """Search abstract, introduction and conclusion in a list of blocks.

        Args:
            blocks (List[str]): The list of blocks.

        Returns:
            Dict[str, Optional[str]]: A dictionary with abstract, intoduction and conclusion.
        """
        ret = {"abstract": None, "introduction": None, "conclusion": None}
        i = 0
        current_section = None
        current_content = ""
        while None in ret.values() and i < len(blocks):
            content = blocks[i]
            words = content.split()
            if self._is_section_title(content, words):
                if current_section is not None:
                    ret[current_section] = current_content
                    current_content = ""
                    current_section = None

                for section in ret.keys():
                    if (
                        ret[section] is None
                        and content.lower().find(section.lower()) >= 0
                    ):
                        current_section = section
                        current_content = ""
            else:
                current_content += " " + content

            i += 1

        return ret

    def start(self) -> Tuple[int, int]:
        """Start the parsing

        Returns:
            Tuple[int, int]: A tuple where the first element indicates how many pdfs
            were parsed and the second how many failures.
        """
        parsed, not_parsed = 0, 0
        for fname in track(
            os.listdir(self.rootdir), description=f"Parsing directory {self.rootdir}..."
        ):
            fpath = os.path.join(self.rootdir, fname)
            if os.path.isfile(fpath) and fname.endswith(".pdf"):
                with fitz.open(fpath) as doc:
                    title, blocks = self._get_title_and_blocks(doc)

                sections = self._search_sections_in_blocks(blocks)
                if None not in sections.values():
                    id = fname[:-4]  # remove '.pdf'
                    self.db.insert_paper(
                        id,
                        title,
                        sections["abstract"],
                        sections["introduction"],
                        sections["conclusion"],
                    )
                    self.parsed_files.append(fpath)
                    parsed += 1
                else:
                    print("Was not able to parse file: " + fname)
                    self.unparsable_files.append(fpath)
                    not_parsed += 1
        return parsed, not_parsed
