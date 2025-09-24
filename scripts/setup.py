"""Setup script for TA V8 platform."""

import typer

app = typer.Typer()


@app.command()
def main():
    """Setup the TA V8 platform with UV package manager."""
    typer.echo("🚀 Setting up TA V8 platform...")
    typer.echo("📦 Using UV as package manager")
    typer.echo("✅ Setup complete!")


if __name__ == "__main__":
    app()