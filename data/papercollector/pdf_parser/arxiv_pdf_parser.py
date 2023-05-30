import os
import re
from typing import Tuple, Optional, List, Dict
import random

import fitz
from rich.progress import track

from papercollector.pdf_parser.base_parser import BasePdfParser


class ArxivPdfParser(BasePdfParser):
    def _get_blocks(self, doc: fitz.Document) -> List[str]:
        """Get the blocks inside a document.

        Args:
            doc (fitz.Document): The document.

        Returns:
            List[str]: The list of blocks.
        """
        blocks = []
        for page in doc:
            for block in page.get_text("blocks"):
                content = block[4]
                content = re.sub(r"-\n", "", content)
                content = re.sub(r"(?<!\r)\r(?!\r)", " ", content)
                content = re.sub(r"(?<!\n)\n(?!\n)", " ", content).strip()

                word_list = content.split()
                word_list = content.split()
                if (
                    self._is_section_title(content, word_list)
                    or len(word_list) > self.MIN_BLOCK_LEN_WORDS
                ):
                    blocks.append(content)

        return blocks

    def _search_sections_in_blocks(self, blocks: List[str]) -> Dict[str, Optional[str]]:
        """Search introduction and conclusion in a list of blocks.

        Args:
            blocks (List[str]): The list of blocks.

        Returns:
            Dict[str, Optional[str]]: A dictionary with intoduction and conclusion.
        """
        ret = {"introduction": None, "conclusion": None}
        i = 0
        current_section = None
        current_content = ""
        while None in ret.values() and i < len(blocks):
            content = blocks[i]
            words = content.split()
            if self._is_section_title(content, words):
                if current_section is not None:
                    current_content = current_content.strip()
                    if len(current_content) > 0:
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

    def start(self, limit: Optional[int] = None) -> Tuple[int, int]:
        """Start the parsing

        Args:
            limit (Optional[int], optional): Limit the nomber of papers to parse. Defaults to None.

        Returns:
            Tuple[int, int]: A tuple where the first element indicates how many pdfs
            were parsed and the second how many failures.
        """        
        parsed, not_parsed = 0, 0

        flist = os.listdir(self.rootdir)
        random.shuffle(flist)
        if limit is None:
            limit = len(flist)
        else:
            limit = min(limit, len(flist))

        for i in track(range(limit), description=f"Parsing directory {self.rootdir}..."):
            fname = flist[i]
            fpath = os.path.join(self.rootdir, fname)
            if os.path.isfile(fpath) and fname.endswith(".pdf"):
                with fitz.open(fpath) as doc:
                    blocks = self._get_blocks(doc)

                sections = self._search_sections_in_blocks(blocks)

                # Extract id from file name
                id = re.search("^(.+)v\d+\.pdf$", fname).group(1)
                id = id.replace("_", "/")

                if (
                    sections["introduction"] is not None and len(sections["introduction"]) > 0
                    and sections["conclusion"] is not None and len(sections["conclusion"]) > 0
                ):
                    self.db.complete_paper(
                        id, sections["introduction"], sections["conclusion"]
                    )
                    self.parsed_files.append(fpath)
                    parsed += 1
                else:
                    self.unparsable_files.append(fpath)
                    self.db.mark_unparsable(id)
                    not_parsed += 1
        return parsed, not_parsed
