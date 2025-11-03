import csv

# Chemin vers le fichier CSV
fichier_csv = "data/comptage-velo-donnees-compteurs.csv"

# Chemin vers le fichier TXT de sortie
fichier_txt = "data/comptage-velo-donnees-compteurs_test.csv"

# Lecture et écriture
with open(fichier_csv, newline="", encoding="utf-8") as csvfile:
    lecteur = csv.reader(csvfile)
    lignes = [ligne for _, ligne in zip(range(1000), lecteur)]

with open(fichier_txt, "w", encoding="utf-8") as txtfile:
    for ligne in lignes:
        txtfile.write(",".join(ligne) + "\n")

print("Les 1000 premières lignes ont été extraites et enregistrées.")
