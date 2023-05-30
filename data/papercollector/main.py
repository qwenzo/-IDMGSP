import typer

from papercollector.arxiv import arxiv_app
from papercollector.scigen import scigen_app

app: typer.Typer = typer.Typer(add_completion=False)

app.add_typer(arxiv_app, name="arxiv", help="Collect papers from arXiv")
app.add_typer(scigen_app, name="scigen", help="Collect papers from SCIgen")
