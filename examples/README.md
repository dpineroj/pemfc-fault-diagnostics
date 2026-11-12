# Mapping examples

A mapping tells `python -m hfc run` where a dataset is and what its columns
and units mean.

| file | adapter | for |
|---|---|---|
| `mapping_table.json` | `table` | a CSV-like bulk file (bulk-only mode) |
| `mapping_seville.json` | `seville` | the Seville raw layout, real or a SYNTHETIC export |
| `mapping_canonical.json` | `canonical` | datasets written by `python -m hfc generate` |

Units are checked, and a failed check stops the run: a current grid that
integrates to the stack current only when read as per-segment amps, a voltage
that looks like millivolts or a stack voltage, or a sign convention that
contradicts the mapping are all errors, not warnings.

`time.unit`: `s`, `ms`, `min`, `h` — or give `time.format` (a `strptime`
format; `column` may be a list of columns to join, e.g. date and time).
`current.unit`: `A` (needs `active_area_cm2`), `A/cm2`, `mA/cm2`.
`voltage.unit`: `V` or `mV`; set `n_cells` if the column is a stack voltage.
