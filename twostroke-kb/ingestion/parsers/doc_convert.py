"""Legacy Word (.doc) -> convert to .docx/text first, then parse.

python-docx CANNOT read old binary .doc. Convert via LibreOffice (headless) or antiword.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..types import ParsedDoc
from . import docx_parser


def parse(path: str | Path) -> ParsedDoc:
    """TODO:
        # Option A (LibreOffice):
        subprocess.run(["libreoffice", "--headless", "--convert-to", "docx",
                        "--outdir", tmp, str(path)], check=True)
        return docx_parser.parse(converted)
        # Option B (antiword): text = subprocess.check_output(["antiword", path])
    """
    raise NotImplementedError("TODO: convert legacy .doc then delegate to docx_parser")
