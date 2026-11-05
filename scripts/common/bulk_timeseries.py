"""
Bulk / system-channel time-series reading. Dataset-agnostic half of what used
to be `load_real_data.py`.

Split out of that module in the November 2026 restructure because it is the
part that transfers: the spatial CDM block parser is specific to the Seville
segmented-cell instrument (closed phase), whereas reading a tab-delimited,
European-locale (comma-decimal) bulk channel file with split date/time columns
into a timestamped float frame is the shape of almost any test-station export.

What makes this reusable rather than merely moved:

  - **comma-decimal handling**: `decimal=","`. Getting this wrong does not
    raise, it silently yields object-dtype columns or truncated integers.
  - **split date/time assembly**: `FECHA` + `HORA` joined before parsing,
    with an EXPLICIT format string rather than inferred parsing. Inference on
    day-first dates is the classic silent corruption -- 03/11/2026 parses as
    March 11 under a US-default inference and no error is raised.
  - **whole-second resolution is a property of the source**, not a choice
    here; downstream joins must tolerate it (the closed phase joined CDM to
    bulk by nearest timestamp with a 1 s tolerance for exactly this reason).

The practice this module exists to carry forward is in
`docs/real_data_notes.md`: a derived channel is validated against an
independent one before it is trusted. Reading the file is the easy part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def read_bulk(filepath: str) -> pd.DataFrame:
    """
    Read a bulk/system channel file into a DataFrame with a single parsed
    `timestamp` column (from FECHA+HORA, whole-second resolution) and all
    other columns as float32.

    Assumes the Seville export's conventions: tab-separated, comma decimal,
    and `FECHA`/`HORA` as separate day-first date and time columns. A loader
    for a differently-shaped export should not bend this function -- it should
    sit beside it and share the discipline, not the parsing.
    """
    df = pd.read_csv(filepath, sep="\t", decimal=",", encoding="utf-8")
    timestamp = pd.to_datetime(
        df["FECHA"] + " " + df["HORA"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["FECHA", "HORA"])
    df = df.astype(np.float32)
    df.insert(0, "timestamp", timestamp.astype("datetime64[ns]"))
    return df
