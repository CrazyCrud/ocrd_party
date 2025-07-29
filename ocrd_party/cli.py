import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ocrd_party.recognize import PartyRecognize


@click.command()
@ocrd_cli_options
def ocrd_party_recognize(*args, **kwargs):
    """
    Recognize text lines with the Party OCR engine.
    """
    return ocrd_cli_wrap_processor(PartyRecognize, *args, **kwargs)
