import os
import shutil
import subprocess
import sys
import zipfile

PYTHON_VERSION = "3.12"
PLATFORM = "manylinux2014_x86_64"
FICHIERS = [
    "lambda_function.py",
    "graph.py",
    "config.py",
    "llm.py",
    "sources.py",
    "store.py",
    "notify.py",
]
DOSSIER = "build"
SORTIE = "lambda.zip"


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--target", DOSSIER, *args])


def main():
    shutil.rmtree(DOSSIER, ignore_errors=True)
    if os.path.exists(SORTIE):
        os.remove(SORTIE)
    os.makedirs(DOSSIER)
    pip("sgmllib3k")
    pip(
        "-r", "requirements-lambda.txt",
        "--platform", PLATFORM,
        "--python-version", PYTHON_VERSION,
        "--implementation", "cp",
        "--only-binary=:all:",
        "--upgrade",
    )
    for f in FICHIERS:
        shutil.copy(f, DOSSIER)
    with zipfile.ZipFile(SORTIE, "w", zipfile.ZIP_DEFLATED) as z:
        for racine, dossiers, fichiers in os.walk(DOSSIER):
            dossiers[:] = [d for d in dossiers if d != "__pycache__"]
            for nom in fichiers:
                chemin = os.path.join(racine, nom)
                z.write(chemin, os.path.relpath(chemin, DOSSIER))
    taille = os.path.getsize(SORTIE) / 1024 / 1024
    print(f"{SORTIE} créé : {taille:.1f} Mo")


if __name__ == "__main__":
    main()
