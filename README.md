# Paper Reproduction

This repository provides a single entry point to regenerate all numerical
results and plots used in the accompanying paper.  The authoritative
implementations live under `src/` and are driven through `run.py`.

## Quickstart

```bash
make env        # create conda env
make all        # run full pipeline
```

Individual stages are exposed as:

```bash
make train
make eval
make plots
```

## Figure and Table Mapping

| Paper label | Generated artifact | Command |
|-------------|-------------------|---------|
| `fig:duffing_spectrum_improved` | `figures/duffing_spectrum_improved.png` | `python run.py plots` |
| `fig:allan_deviation_improved` | `figures/allan_deviation_improved.png` | `python run.py plots` |
| `fig:pnc_band_structure` | `figures/bands_pwe_improved.png` | `python run.py plots` |
| `fig:bpsk_ber` | `figures/ber_rrc_bpsk.png` | `python run.py plots` |
| `fig:s21_f` | `figures/fdtd_S21.png` | `python run.py plots` |
| `fig:fdtd_snapshot` | `figures/fdtd_snapshot_improved.png` | `python run.py plots` |
| `fig:ssh_edge_modes` | `figures/ssh_modes_improved.png` | `python run.py plots` |

See `PAPER_MAP.md` for a more exhaustive mapping.
