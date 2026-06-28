# PLS Studio

Desktop app Python/PySide6 for visual PLS-SEM analysis. This project is positioned as an independent PLS-SEM research workbench, not a SmartPLS clone.

## Run

```powershell
cd "D:\Huyền Mai\PySmartPLS"
.\venv\Scripts\python.exe main.py
```

If you recreate the environment:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

## Current MVP Features

- Project workspace: new/open/save `.plsproj`, export/import backup `.zip`.
- Data import: CSV, TXT, XLS, XLSX, SAV.
- Data view: preview, descriptive statistics, missing-value warnings, non-numeric warnings, scale hints, cleaned-data export.
- Model canvas: latent constructs, indicators, directed paths, comments, drag/drop indicators, context menu rename/delete/reflective-formative, zoom, image export.
- Model checker: missing indicators, unknown constructs, duplicate assignments, directed-cycle prevention.
- Analysis engine: PLS-SEM algorithm, Mode A reflective, Mode B formative, path/factor/centroid weighting, sum scores/OLS.
- Results: path coefficients, R2, adjusted R2, f2, inner VIF, outer loadings/weights, Cronbach alpha, rho_A approximation, composite reliability, AVE, Fornell-Larcker, HTMT, cross-loadings, total/indirect effects, approximate SRMR.
- Bootstrapping: fixed seed, configurable subsamples, percentile confidence interval, t and p values.
- Export: Excel workbook and HTML report.
- Sample project for quick testing.

## Planned Next Modules

- Blindfolding Q2.
- PLSpredict.
- MGA/MICOM.
- IPMA.
- Word/PDF thesis-ready report.
- Vietnamese/English UI switch and richer interpretation text.
