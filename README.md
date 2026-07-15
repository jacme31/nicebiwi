# Biwipy Proto (NiceGUI)

[English](README.md) | [Francais](README.fr.md)

Minimal GUI prototype to quickly run and test `biwipy`.

## Recommended environment

The recommended way to run this prototype is to use a conda environment such as `biwipy-pypi`, where `biwipy` and `nicegui` are already installed.

```powershell
conda activate biwipy-pypi
python nicebiwi.py
```

In VS Code, also select the Python interpreter from this environment to avoid false import warnings.

Notes:
- [nicebiwi.py](nicebiwi.py) includes a development fallback that can import the neighboring workspace `../biwipy` if needed.
- This fallback is not the target execution path. The `biwipy-pypi` environment remains the recommended launch setup.

## Local alternative

If you do not want to use the shared conda environment, you can create a local environment for this prototype.

## 1) Create the environment (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Install dependencies

```powershell
pip install -r requirements.txt
pip install -e ../biwipy
```

Notes:
- `biwipy` depends on `pygrib`.
- On Windows, `pygrib` is often easier to manage with conda-forge.

## 3) Run the prototype

```powershell
python nicebiwi.py
```

Then open: <http://127.0.0.1:8081>

## Prototype features

- Cyclist profile management
- Weather model selection (GFS, IFS, or none)
- GPX file upload
- Simulation and Replay (if GPX contains timestamps)
- Basic stats display (distance, timestamps, duration)
- Optional charts and interactive map

## UI internationalization

- UI translations are externalized in [i18n/fr.json](i18n/fr.json) and [i18n/en.json](i18n/en.json)
- Code loads these files at startup from [nicebiwi.py](nicebiwi.py)
