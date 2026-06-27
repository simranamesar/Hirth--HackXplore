"""Spreadsheets (.xlsx/.csv) -> TABLE-AWARE. Preserve rows/cols/units/formulas.

Critical: write each value to the `structured_facts` table so spec_lookup returns
EXACT numbers. Do NOT flatten a calc sheet into prose and lose the values.
"""
from __future__ import annotations

from pathlib import Path

from ..types import ParsedDoc, Table


def parse(path: str | Path) -> ParsedDoc:
    """TODO:
        import pandas as pd / openpyxl
        - read each sheet; keep header row as column labels
        - detect units (e.g. column "Temp [°C]") -> Table.units
        - emit Table objects AND structured_facts rows (done in knowledge_base.store)
        - also build a short text summary per sheet for semantic recall
    Examples in scope: 'Berechnung Schallgeschwindigkeit im Auspuff.xlsx',
                       'Fuel_Kraftstoffe_Übersicht_Daten.xlsx'
    """
    raise NotImplementedError("TODO: table-aware spreadsheet parsing")
