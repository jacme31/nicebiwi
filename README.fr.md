# Biwipy Proto (NiceGUI)

[English](README.md) | [Francais](README.fr.md)

Prototype minimal d'interface graphique pour utiliser et tester `biwipy` rapidement.

## Environnement recommande

Le mode recommande pour ce proto est d'utiliser un environnement conda, par exemple `biwipy-pypi`, dans lequel `biwipy` et `nicegui` sont deja installes.

```powershell
conda activate biwipy-pypi
python nicebiwi.py
```

Dans VS Code, selectionner aussi l'interpreteur Python de cet environnement pour eviter les faux warnings d'import.

Notes:
- [nicebiwi.py](nicebiwi.py) contient un fallback de developpement qui peut importer le workspace voisin `../biwipy` si besoin.
- Ce fallback n'est pas le chemin d'execution cible. L'environnement `biwipy-pypi` reste la reference de lancement recommandee.

## Alternative locale

Si tu ne veux pas utiliser l'environnement conda partage, tu peux aussi creer un environnement local pour le proto.

## 1) Creation de l'environnement (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Installation des dependances

```powershell
pip install -r requirements.txt
pip install -e ../biwipy
```

Notes:
- `biwipy` depend de `pygrib`.
- Sur Windows, `pygrib` se gere souvent plus simplement via conda-forge.

## 3) Lancer le proto

```powershell
python nicebiwi.py
```

Puis ouvrir: <http://127.0.0.1:8081>

## Fonctionnalites du proto

- Gestion de profils cyclistes
- Choix d'un modele meteo (GFS, IFS ou aucun)
- Upload d'un fichier GPX
- Simulation et Replay (si le GPX contient des timestamps)
- Affichage des stats de base (distance, timestamps, duree)
- Graphiques optionnels et carte interactive

## Internationalisation UI

- Les traductions UI sont externalisees dans [i18n/fr.json](i18n/fr.json) et [i18n/en.json](i18n/en.json)
- Le code charge ces fichiers au demarrage depuis [nicebiwi.py](nicebiwi.py)
