"""
Loader for the real PEMFC current-density/temperature-mapping dataset:

Toharias, B., Suarez, C., Iranzo, A., Salva, M., Rosa, F. (2024), "Dataset and
measurements from a current density sensor during experimental testing of
dynamic load cycling for a parallel-serpentine design of a proton exchange
membrane fuel cell," Data in Brief.
https://www.sciencedirect.com/science/article/pii/S2352340924003615

File-format findings (see Task 1 audit, data/raw is not tracked in git):
  - CDM_C_*.dat / CDM_T_*.dat are repeating per-timestep BLOCKS, not flat
    tables: human timestamp, LabVIEW-epoch timestamp, blank, a field label,
    (CDM_C only: a scalar "voltage" value, blank, a second label), then an
    N x M tab-delimited comma-decimal grid, then a blank separator line.
    CDM_C default: 26 lines/block, 18x18 current-density grid (A/cm^2).
    CDM_T default: 14 lines/block, 9x9 temperature grid (degC).
  - The bulk/system file (PC_*.dat or FC-DLC_*.dat) is a conventional flat
    table: one tab-delimited header row, then one row per second, with
    FECHA (date) + HORA (time, whole-second resolution) and comma-decimal
    numeric columns including INTENSIDAD (the load-current setpoint/reading).
  - CDM_C and CDM_T are emitted in lockstep (identical timestamps, same
    block count) and can be joined by row index. The bulk file runs on its
    own clock (different start/end time, whole-second resolution, different
    row count) and must be joined to the CDM data by nearest timestamp.
  - The first several minutes of a run are a DAQ warm-up transient
    (irregular sampling interval, anomalous near-zero current values) and
    are excluded by default via `warmup_seconds`.
  - CDM_C's "voltage" scalar and the bulk file's V001-V007 channels are NOT
    known to be the same physical signal (magnitudes disagree). Both are
    loaded as separate, clearly-labeled fields; no reconciliation is
    attempted here.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "PEMFC_Parallel-Serpentine")

CDM_C_BLOCK_SIZE = 26
CDM_C_GRID_SHAPE = (18, 18)
CDM_T_BLOCK_SIZE = 14
CDM_T_GRID_SHAPE = (9, 9)


def _parse_human_timestamp(line: str) -> datetime:
    """Parse a block timestamp line 'DD/MM/YY HH:MM:SS,mmm' into a datetime."""
    date_part, time_part = line.strip().split(" ")
    hms, ms = time_part.split(",")
    dt = datetime.strptime(f"{date_part} {hms}", "%d/%m/%y %H:%M:%S")
    return dt + timedelta(milliseconds=int(ms))


def parse_cdm_blocks(
    filepath: str,
    block_size: int = CDM_C_BLOCK_SIZE,
    grid_shape: tuple[int, int] = CDM_C_GRID_SHAPE,
) -> dict:
    """
    Parse a CDM_C/CDM_T block-formatted .dat file.

    Each block is `block_size` lines: a human-readable timestamp, a
    LabVIEW-epoch timestamp (seconds since 1904-01-01, comma decimal), a
    blank line, a field label, an optional scalar sub-block (label + value +
    blank + label, present only in CDM_C's "voltage" field), the
    `grid_shape` grid (tab-delimited, comma decimal), and a trailing blank
    separator line. Scalar presence is auto-detected per block from the
    field-count of the line immediately after the first label, not assumed
    from `block_size`.

    Parameters
    ----------
    filepath : path to the CDM_C_*.dat or CDM_T_*.dat file
    block_size : total lines per block (default: confirmed CDM_C value, 26)
    grid_shape : (rows, cols) of the spatial grid (default: confirmed CDM_C
        value, (18, 18))

    Returns
    -------
    dict with:
        'timestamps' : np.ndarray[datetime64[ms]], shape (n_blocks,)
        'labview_timestamps' : np.ndarray[float64], shape (n_blocks,)
            Raw LabVIEW-epoch seconds, kept only as a cross-check against
            'timestamps' -- not converted to real dates here.
        'grid' : np.ndarray[float32], shape (n_blocks, *grid_shape)
        'scalar' : np.ndarray[float32] shape (n_blocks,), or None if no
            block in the file carries a scalar sub-block (e.g. CDM_T).

    Raises
    ------
    ValueError if the file's line count is not a multiple of `block_size`,
    if a block's structure doesn't match the expected layout, or if the
    resulting grid array's shape doesn't match `grid_shape`.
    """
    grid_rows, grid_cols = grid_shape

    with open(filepath, "rb") as f:
        total_lines = 0
        while True:
            data = f.read(1 << 20)
            if not data:
                break
            total_lines += data.count(b"\n")

    if total_lines % block_size != 0:
        raise ValueError(
            f"{filepath}: {total_lines} lines is not a multiple of "
            f"block_size={block_size}; cannot safely chunk into blocks."
        )
    n_blocks = total_lines // block_size

    timestamps: list[datetime] = []
    labview_ts: list[float] = []
    scalars: list[float] = []
    has_scalar: Optional[bool] = None
    grid = np.empty((n_blocks, grid_rows, grid_cols), dtype=np.float32)

    with open(filepath, "r", encoding="utf-8") as f:
        for block_idx in range(n_blocks):
            block = [f.readline() for _ in range(block_size)]
            if not block[-1]:
                raise ValueError(
                    f"{filepath}: unexpected EOF while reading block {block_idx}"
                )

            timestamps.append(_parse_human_timestamp(block[0]))
            labview_ts.append(float(block[1].strip().replace(",", ".")))

            probe_fields = block[4].strip().split("\t")
            block_has_scalar = len(probe_fields) == 1 and len(block[4].strip()) > 0
            if has_scalar is None:
                has_scalar = block_has_scalar
            elif has_scalar != block_has_scalar:
                raise ValueError(
                    f"{filepath}: block {block_idx} scalar presence "
                    f"({block_has_scalar}) disagrees with earlier blocks "
                    f"({has_scalar}) -- inconsistent block structure."
                )

            if has_scalar:
                scalars.append(float(probe_fields[0].replace(",", ".")))
                grid_start = 7  # timestamp(2) + blank + label + scalar + blank + label
            else:
                grid_start = 4  # timestamp(2) + blank + label

            expected_block_size = grid_start + grid_rows + 1  # + trailing blank
            if expected_block_size != block_size:
                raise ValueError(
                    f"{filepath}: block {block_idx} implies block_size="
                    f"{expected_block_size} from its structure, but "
                    f"block_size={block_size} was requested. Grid reshape "
                    f"would be silently wrong -- refusing to continue."
                )

            for r in range(grid_rows):
                row_line = block[grid_start + r].replace(",", ".")
                values = np.fromstring(row_line, sep="\t", dtype=np.float32)
                if values.shape[0] != grid_cols:
                    raise ValueError(
                        f"{filepath}: block {block_idx} grid row {r} has "
                        f"{values.shape[0]} values, expected {grid_cols}."
                    )
                grid[block_idx, r, :] = values

            trailing = block[grid_start + grid_rows].strip()
            if trailing != "":
                raise ValueError(
                    f"{filepath}: block {block_idx} expected a blank "
                    f"separator line after the grid, got: {trailing!r}"
                )

    if grid.shape[1:] != grid_shape:
        raise ValueError(
            f"{filepath}: parsed grid shape {grid.shape[1:]} does not match "
            f"requested grid_shape {grid_shape}."
        )

    return {
        "timestamps": np.array(timestamps, dtype="datetime64[ms]"),
        "labview_timestamps": np.array(labview_ts, dtype=np.float64),
        "grid": grid,
        "scalar": np.array(scalars, dtype=np.float32) if has_scalar else None,
    }


def read_bulk(filepath: str) -> pd.DataFrame:
    """
    Read a bulk/system channel file (PC_*.dat or FC-DLC_*.dat) into a
    DataFrame with a single parsed `timestamp` column (from FECHA+HORA,
    whole-second resolution) and all other columns as float32.
    """
    df = pd.read_csv(filepath, sep="\t", decimal=",", encoding="utf-8")
    timestamp = pd.to_datetime(
        df["FECHA"] + " " + df["HORA"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["FECHA", "HORA"])
    df = df.astype(np.float32)
    df.insert(0, "timestamp", timestamp.astype("datetime64[ns]"))
    return df


def _find_one(pattern: str, description: str) -> str:
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {description} matching {pattern!r}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def load_run(
    config: str,
    run_type: str = "FC-DLC",
    data_root: str = DATA_ROOT,
    warmup_seconds: float = 300.0,
    merge_tolerance: str = "1s",
) -> dict:
    """
    Load one (gas_config, run_type) pair into a single aligned structure.

    Parameters
    ----------
    config : one of "Normal_Flow", "Inverse_Flow", "Inverse_Air_Flow",
        "Inverse_Hydrogen_Flow"
    run_type : "FC-DLC" (dynamic load cycle) or "PC" (polarization curve)
    warmup_seconds : seconds of DAQ warm-up transient to drop from the start
        of the run (see module docstring). Set to 0 to keep the full run.
    merge_tolerance : pandas Timedelta-parseable string; max gap allowed
        when matching a CDM timestep to the nearest bulk-file row.

    Returns
    -------
    dict with:
        'config', 'run_type' : str
        'timestamp' : np.ndarray[datetime64[ms]], shape (T,) -- CDM clock
        'current_grid' : np.ndarray[float32], shape (T, 18, 18), A/cm^2
        'temp_grid' : np.ndarray[float32], shape (T, 9, 9), degC
        'bulk' : pd.DataFrame, shape (T, ...) -- bulk channels + CDM scalar
            'voltage_cdm' + raw LabVIEW timestamps, merge_asof-aligned to
            'timestamp'; bulk columns are NaN for any row that had no match
            within `merge_tolerance`
        'n_rows_before_warmup_exclusion', 'n_rows_after_warmup_exclusion' : int
        'n_unmatched_bulk' : int -- CDM rows with no bulk match within tolerance
    """
    run_dir = _find_one(
        os.path.join(data_root, config, f"*_{run_type}_*"),
        f"{run_type} run directory under {config}",
    )
    cdm_c_path = _find_one(os.path.join(run_dir, "CDM_C_*.dat"), "CDM_C file")
    cdm_t_path = _find_one(os.path.join(run_dir, "CDM_T_*.dat"), "CDM_T file")
    bulk_path = _find_one(os.path.join(run_dir, f"{run_type}_*.dat"), "bulk file")

    cdm_c = parse_cdm_blocks(cdm_c_path, CDM_C_BLOCK_SIZE, CDM_C_GRID_SHAPE)
    cdm_t = parse_cdm_blocks(cdm_t_path, CDM_T_BLOCK_SIZE, CDM_T_GRID_SHAPE)
    bulk = read_bulk(bulk_path)

    if cdm_c["grid"].shape[0] != cdm_t["grid"].shape[0]:
        raise ValueError(
            f"{config}/{run_type}: CDM_C has {cdm_c['grid'].shape[0]} "
            f"blocks but CDM_T has {cdm_t['grid'].shape[0]} -- cannot join "
            f"by row index."
        )
    if not np.array_equal(cdm_c["timestamps"], cdm_t["timestamps"]):
        n_diff = int(np.sum(cdm_c["timestamps"] != cdm_t["timestamps"]))
        raise ValueError(
            f"{config}/{run_type}: CDM_C and CDM_T timestamps disagree on "
            f"{n_diff} of {len(cdm_c['timestamps'])} blocks -- refusing to "
            f"join by row index."
        )

    ts_int = cdm_c["timestamps"].astype("datetime64[ms]").astype(np.int64)
    if np.any(np.diff(ts_int) < 0):
        raise ValueError(
            f"{config}/{run_type}: CDM timestamps are not monotonically "
            f"non-decreasing -- cannot safely merge_asof against bulk data."
        )

    cdm_index = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(cdm_c["timestamps"]).astype("datetime64[ns]"),
            "labview_ts_current": cdm_c["labview_timestamps"],
            "labview_ts_temperature": cdm_t["labview_timestamps"],
            "voltage_cdm": cdm_c["scalar"],
        }
    )

    merged = pd.merge_asof(
        cdm_index,
        bulk.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(merge_tolerance),
    )
    bulk_probe_col = "INTENSIDAD"
    n_unmatched = int(merged[bulk_probe_col].isna().sum())
    if n_unmatched:
        print(
            f"[load_run] {config}/{run_type}: {n_unmatched}/{len(merged)} "
            f"CDM rows had no bulk-file match within tolerance="
            f"{merge_tolerance}"
        )

    n_before = len(merged)
    if warmup_seconds > 0:
        cutoff = merged["timestamp"].iloc[0] + pd.Timedelta(seconds=warmup_seconds)
        keep_mask = (merged["timestamp"] >= cutoff).to_numpy()
    else:
        keep_mask = np.ones(n_before, dtype=bool)

    current_grid = cdm_c["grid"][keep_mask]
    temp_grid = cdm_t["grid"][keep_mask]
    merged = merged.loc[keep_mask].reset_index(drop=True)
    n_after = len(merged)

    if current_grid.shape[1:] != CDM_C_GRID_SHAPE:
        raise ValueError(
            f"{config}/{run_type}: current_grid shape {current_grid.shape} "
            f"does not match expected {CDM_C_GRID_SHAPE} after warm-up "
            f"exclusion."
        )
    if temp_grid.shape[1:] != CDM_T_GRID_SHAPE:
        raise ValueError(
            f"{config}/{run_type}: temp_grid shape {temp_grid.shape} does "
            f"not match expected {CDM_T_GRID_SHAPE} after warm-up exclusion."
        )

    return {
        "config": config,
        "run_type": run_type,
        "timestamp": merged["timestamp"].to_numpy(),
        "current_grid": current_grid,
        "temp_grid": temp_grid,
        "bulk": merged,
        "n_rows_before_warmup_exclusion": n_before,
        "n_rows_after_warmup_exclusion": n_after,
        "n_unmatched_bulk": n_unmatched,
    }


if __name__ == "__main__":
    configs = [
        "Normal_Flow",
        "Inverse_Flow",
        "Inverse_Air_Flow",
        "Inverse_Hydrogen_Flow",
    ]

    for config in configs:
        run = load_run(config, run_type="FC-DLC")
        ts = run["timestamp"]
        cg = run["current_grid"]
        tg = run["temp_grid"]
        intensidad = run["bulk"]["INTENSIDAD"].to_numpy()

        print(f"\n=== {config} / FC-DLC ===")
        print(
            f"rows: {run['n_rows_before_warmup_exclusion']} before warm-up "
            f"exclusion -> {run['n_rows_after_warmup_exclusion']} after"
        )
        print(f"timestamp range: {ts[0]} -> {ts[-1]}")
        print(f"unmatched bulk rows (outside merge tolerance): {run['n_unmatched_bulk']}")
        print(
            f"current_grid  A/cm^2: min={cg.min():.4f} max={cg.max():.4f} "
            f"mean={cg.mean():.4f}"
        )
        print(
            f"temp_grid     degC  : min={tg.min():.2f} max={tg.max():.2f} "
            f"mean={tg.mean():.2f}"
        )
        n_nan = int(np.isnan(intensidad).sum())
        i_min, i_max, i_std = (
            np.nanmin(intensidad),
            np.nanmax(intensidad),
            np.nanstd(intensidad),
        )
        print(
            f"INTENSIDAD    A     : min={i_min:.3f} max={i_max:.3f} "
            f"std={i_std:.3f} ({n_nan} NaN from unmatched merges) "
            f"({'varying -- looks like a real load cycle' if i_std > 0.5 else 'WARNING: looks nearly constant'})"
        )
