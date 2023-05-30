# Papercollector

Generate fake papers and download real ones.

## Table of content

- [Papercollector](#papercollector)
  - [Table of content](#table-of-content)
  - [System requirements](#system-requirements)
  - [Basic usage](#basic-usage)
  - [Guides](#guides)
    - [Python dependencies and virtual environment with Poetry](#python-dependencies-and-virtual-environment-with-poetry)
    - [Collecting arXiv's papers](#collecting-arxivs-papers)
      - [Reserving and exporting papers](#reserving-and-exporting-papers)
    - [Collecting SCIgen's papers](#collecting-scigens-papers)
    - [Accessing data from Python Pandas](#accessing-data-from-python-pandas)
  - [Databases structure](#databases-structure)
    - [arXiv database (papers table)](#arxiv-database-papers-table)
    - [SCIgen database (papers table)](#scigen-database-papers-table)
  
## System requirements

- [Python](https://www.python.org/)
- [Poetry](https://python-poetry.org/) for dependencies management
- [Docker](https://www.docker.com/) to run SCIgen
- [gsutil](https://cloud.google.com/storage/docs/gsutil_install) to download arXiv's pdfs

## Basic usage

1. Install the dependencies and activate the virtual environment.
2. Use `python -m papercollector --help` to see the aviable commands.

## Guides

### Python dependencies and virtual environment with Poetry

See [Poetry documentation](https://python-poetry.org/docs/basic-usage/).

In short use `poetry install` to install the dependencies and then select the virtual environment from your IDE.

### Collecting arXiv's papers

1. Download the metadata from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) and place it in the `data` folder.
2. Parse the metadata using `python -m papercollector arxiv parse-metadata`
3. Download papers pdfs using `python -m papercollector arxiv download-pdfs`. To filter categories use `python -m papercollector arxiv download-pdfs --categories "cs.CV cs.CL"` or `python -m papercollector arxiv download-pdfs --categories "cs. math."`. It is also possible to filter by year using `--year-from` and `--year-to`, both are inclusive.
4. Start parsing pdfs with `python -m papercollector arxiv parse-pdfs`

The database will be at `papers_db/arxiv.db`.

#### Reserving and exporting papers
To reserve some papers for a destination the command `arxiv reserve` can be used. After reserving the papers they can be exported to CSV using the `arxiv export-csv` command.

### Collecting SCIgen's papers

1. Run the Docker engine
2. Generate the pdfs with `python -m papercollector scigen generate-pdfs`
3. Parse the pdfs with `python -m papercollector scigen parse-pdfs`

The database will be at `papers_db/scigen.db`.

### Accessing data from Python Pandas

SQLite allows to perform queries on the data for ease of management.

To access the data from Pandas:

```python
import sqlite3
import pandas as pd

con = sqlite3.connect("data/papers/arxiv.db")
df = pd.read_sql_query("""
        SELECT title, abstract, introduction, conclusion FROM papers
        WHERE is_pdf_parsed = 1
        AND categories LIKE '%cs.%'
    """, con)

con.close()
```

## Databases structure

A good program to browse the database is [DB Browser for SQLite](https://sqlitebrowser.org/).

### arXiv database (papers table)

| Field Name    | Type             | Description                                                                                                               |
| ------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| id            | TEXT PRIMARY KEY | The arXiv id                                                                                                              |
| year          | INTEGER NOT NULL | The year                                                                                                                  |
| destination | TEXT | Indicates the destination of the paper. Only the `real` destination is set automatically when a paper is parsed, other destinations must be specified using the `reserve` command. |
| is_pdf_parsed | INTEGER NOT NULL | Integer, indicates whether the pdf has been parsed. `0` parsing not attempted, `-1` parsing failed, `1` parsing succeeded |
| doi           | TEXT             | The DOI                                                                                                                   |
| last_version  | TEXT NOT NULL    | Indicates the last version                                                                                                |
| title         | TEXT NOT NULL    | The title                                                                                                                 |
| categories    | TEXT             | Space-separated list of arXiv categories                                                                                  |
| abstract      | TEXT NOT NULL    | The abstract                                                                                                              |
| introduction  | TEXT             | The introduction, present only if `is_pdf_parsed=1`                                                                       |
| conclusion    | TEXT             | The conclusion, present only if `is_pdf_parsed=1`                                                                         |

### SCIgen database (papers table)

| Field Name   | Type             | Description                                       |
| ------------ | ---------------- | ------------------------------------------------- |
| id           | TEXT PRIMARY KEY | It is the MD5 checksum and filename of the paper. |
| title        | TEXT NOT NULL    | The title                                         |
| abstract     | TEXT NOT NULL    | The abstract                                      |
| introduction | TEXT NOT NULL    | The introduction                                  |
| conclusion   | TEXT NOT NULL    | The conclusion                                    |
