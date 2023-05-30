import typer
import subprocess
from pathlib import Path
import hashlib
import os

from rich import print
from rich.progress import track, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import pandas as pd

from papercollector.db import BaseDB
from papercollector.pdf_parser import ScigenPdfParser

scigen_app: typer.Typer = typer.Typer(add_completion=False)


@scigen_app.command()
def generate_pdfs(
    n: int = typer.Option(
        ...,
        help="How many papers to generate",
        prompt="How many papers do you want to generate?",
        min=1,
    ),
    output: Path = typer.Option(
        Path("./papers_pdf/scigen"),
        help="The directory where to store the generated pdfs",
        exists=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Generate papers with SCIgen"""
    file = output / "paper.pdf"

    # Create container
    subprocess.run(
            [
                "docker",
                "run",
                "-it",
                "-v",
                str(output) + ":/opt/scigen/out/",
                "--name",
                "scigen"
                "soerface/scigen",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    for _ in track(range(n), description="Generating papers..."):
        subprocess.run(
            [
                "docker",
                "start",
                "-i",
                "scigen"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.isfile(file):
            print(
                "[bold red]There was an error generating the paper.[/bold red] [red]Is the Docker engine running?[/red]"
            )
            raise typer.Exit(code=1)

        checksum = hashlib.md5(open(file, "rb").read()).hexdigest()
        os.rename(file, output / (checksum + ".pdf"))
    print("[bold green]Success![/bold green] :smiley:")


@scigen_app.command()
def parse_pdfs(
    input: Path = typer.Option(
        Path("./papers_pdf/scigen"),
        help="The directory where the generated pdfs are stored",
        exists=True,
        file_okay=True,
        writable=True,
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
    """Parse the pdfs generated with SCIgen"""
    db = BaseDB(db_dir, "scigen")
    parser = ScigenPdfParser(input, db)
    parsed, not_parsed = parser.start()
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

@scigen_app.command()
def export_csv(
    output: Path = typer.Option(
        Path("./papers_csv/scigen.csv"),
        help="The output CSV file",
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
            description="Export papers to CSV...",
            total=None,
        )
        db = BaseDB(db_dir, "scigen")

        papers_df: pd.DataFrame = db.get_papers_df()
        papers_df.to_csv(output, index = False)

    print("[bold green]Success![/bold green] :smiley:")