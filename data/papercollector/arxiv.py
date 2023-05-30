from pathlib import Path
import json
import subprocess
from shutil import which
import datetime
import re
import os
from hashlib import md5
from base64 import b64encode
from typing import Optional

import typer
import pandas as pd
from rich import print
from rich.progress import track, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from papercollector.db import ArxivDB
from papercollector.pdf_parser import ArxivPdfParser


arxiv_app: typer.Typer = typer.Typer(add_completion=False)


@arxiv_app.command()
def test():
    db = ArxivDB(Path("./papers_db"))
    db.create_indexes()


@arxiv_app.command()
def parse_metadata(
    file: Path = typer.Argument(
        Path("./arxiv-metadata-oai-snapshot.json"),
        help="The path to the JSON file containing the metadata",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True,
    ),
    db_dir: Path = typer.Option(
        Path("./papers_db"),
        help="The directory where to store the database with the metadata",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Parse the arXiv JSON containing the metadata downloaded from https://www.kaggle.com/datasets/Cornell-University/arxiv"""
    db = ArxivDB(db_dir)
    cur_year = datetime.date.today().year

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            description="Starting metadata parse from json (this may take a while)...",
            total=None,
        )
        with open(file, "rb") as f:
            for line in f:
                paper: dict = json.loads(line)

                # https://regex101.com/r/VCm62o/1
                paper["title"] = re.sub(" *\n *|  +", " ", paper["title"]).strip()
                paper["abstract"] = re.sub(" *\n *|  +", " ", paper["abstract"]).strip()

                # Extract year and month from id
                if "/" in paper["id"]:
                    tmp = paper["id"].split("/")[1]
                else:
                    tmp = paper["id"]
                paper_date = datetime.datetime.strptime(tmp[:4], "%y%m")
                if paper_date.year > cur_year:
                    paper_date.replace(year=paper_date.year - 100)

                db.insert_paper(
                    paper["id"],
                    paper_date.year,
                    paper["doi"],
                    paper["versions"][-1]["version"],
                    paper["title"],
                    paper["categories"],
                    paper["abstract"],
                )
    db.commit()
    print("[bold green]Success![/bold green] :smiley:")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            description="Creating SQL indexes for speeed...",
            total=None,
        )
        db.create_indexes()

    db.commit_and_close()
    print("[bold green]Success![/bold green] :smiley:")


@arxiv_app.command()
def download_pdfs(
    n: int = typer.Option(
        ...,
        prompt="How many papers do you want to download?",
        help="Number of papers to download",
        min=1,
    ),
    year_from: int = typer.Option(
        None,
        help="Get papers after a certain year (included)",
        min=0,
    ),
    year_to: int = typer.Option(
        None,
        help="Get papers before a certain year (included)",
        min=1990,
    ),
    categories: str = typer.Option(
        None,
        help="Filter by categories (space separated list)",
    ),
    skip_unparsable: bool = typer.Option(
        True,
        help="If True skip unparsable papers",
    ),
    skip_parsed: bool = typer.Option(
        True,
        help="If True skip parsed papers",
    ),
    output: Path = typer.Option(
        Path("./papers_pdf/arxiv"),
        help="The path to the directory containing the papers in pdf format",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True,
    ),
    db_dir: Path = typer.Option(
        Path("./papers_db"),
        help="The directory where to store the database",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Download papers pdfs from arXiv"""
    if year_from is not None and year_to is not None and year_from > year_to:
        raise typer.BadParameter("Option year-from must be smaller than option year-to")

    gsutil = which("gsutil")
    if gsutil is None:
        print("[bold red]Error: cannot find gsutil in PATH[/bold red]")
        raise typer.Exit(code=1)

    # Clear old report
    report_path = output / "report.csv.tmp"
    open(report_path, "w").close()

    db = ArxivDB(db_dir)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            description="Generating download urls...",
            total=None,
        )
        urls = db.papers_to_download(
            number=n,
            year_from=year_from,
            year_to=year_to,
            categories=categories,
            skip_unparsable=skip_unparsable,
            skip_parsed=skip_parsed,
        )

    if len(urls) < n:
        print(
            f"There are only {len(urls)} papers which are not parsed yet and respects the filters"
        )

    print(f"Staring download of {len(urls)} papers...")
    urls_path = output / "urls.txt"
    with open(urls_path, "w") as f:
        f.writelines("\n".join(urls))
    with open(urls_path, "rb") as f:
        subprocess.run(
            [gsutil, "-m", "cp", "-n", "-r", "-I", "-L", report_path, output], stdin=f
        )
    os.remove(urls_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            description="Renaming some files...",
            total=None,
        )
        # Open report and delete file
        report = pd.read_csv(report_path)
        os.remove(report_path)

        for _, row in report.iterrows():
            res = re.search(
                r"^gs:\/\/arxiv-dataset\/arxiv\/(.+)\/pdf\/\d*\/(.+)\.pdf$",
                row["Source"],
            )
            category, fname = res.group(1), res.group(2)
            fpath = output / (fname + ".pdf")
            if category != "arxiv" and os.path.isfile(fpath):
                checksum = str(
                    b64encode(md5(open(fpath, "rb").read()).digest()), encoding="ascii"
                )
                if row["Md5"] == checksum:
                    new_fpath = output / f"{category}_{fname}.pdf"
                    if not os.path.isfile(fpath):
                        os.rename(fpath, new_fpath)

    print("[bold green]Success![/bold green] :smiley:")


@arxiv_app.command()
def count_parsed_papers(
    db_dir: Path = typer.Argument(
        Path("./papers_db"),
        help="The directory where to store the database",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    )
) -> None:
    """Count the number of parsed papers in the database."""
    db = ArxivDB(db_dir)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            description="Counting parsed papers in the database...",
            total=None,
        )
        n_parsed = db.count_parsed_papers()
        db.close()
    print(f"In the database there are [bold]{str(n_parsed)}[/bold] parsed papers.")


@arxiv_app.command()
def parse_pdfs(
    input: Path = typer.Argument(
        Path("./papers_pdf/arxiv"),
        help="The path to the directory containing the papers in pdf format",
        exists=True,
        file_okay=True,
        readable=True,
        resolve_path=True,
    ),
    limit: Optional[int] = typer.Option(
        None, help="Limt the parsing to a number of pdfs", min=1
    ),
    db_dir: Path = typer.Option(
        Path("./papers_db"),
        help="The directory where to store the database",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Parse the downloaded pdfs to extract introduction and conclusion"""
    db = ArxivDB(db_dir)
    parser = ArxivPdfParser(input, db)
    parsed, not_parsed = parser.start(limit)
    db.commit_and_close()
    print("[bold green]Success![/bold green] :smiley:")
    print(f"{parsed} where correctly parsed while {not_parsed} were not parsed.")

    delete_parsed = typer.confirm(
        "Do you want to delete the parsed pdfs?", default=True
    )
    if delete_parsed:
        parser.delete_parsed()
        print("[bold green]Done![/bold green] :smiley:")

    delete_unparsable = typer.confirm(
        "Do you want to delete the unparsable pdfs?", default=True
    )
    if delete_unparsable:
        parser.delete_unparsable()
        print("[bold green]Done![/bold green] :smiley:")


@arxiv_app.command()
def reserve(
    destination: str = typer.Option(
        ...,
        help="The destination of the papers"
    ),
    n: int = typer.Option(
        ...,
        help="How many papers to reserve for the destination"
    ),
    db_dir: Path = typer.Option(
        Path("./papers_db"),
        help="The directory where to store the database with the metadata",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Reserve some papers for a destination"""
    db = ArxivDB(db_dir)
    db.reserve_for_destination(destination, n)


@arxiv_app.command()
def export_csv(
    output: Path = typer.Option(
        Path("./papers_csv"),
        help="The directory where to store the CSV files",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    db_dir: Path = typer.Option(
        Path("./papers_db"),
        help="The directory where to store the database with the metadata",
        exists=True,
        file_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Export the downloaded papers to CSV."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task(
            description="Getting list of destinations...",
            total=None,
        )

        db = ArxivDB(db_dir)
        destinations = db.get_destinations()

    for d in track(destinations, description="Generating CSV files, this may take a while..."):
        papers_df: pd.DataFrame = db.get_papers_df_with_destination(d)
        papers_df.to_csv(output / (d + ".csv"), index = False)

    print("[bold green]Success![/bold green] :smiley:")
