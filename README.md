# [Titre]

## À propos

Ce projet a pour objectif de déployer un modèle de classification pour prévoir les démissions à partir d'un jeux de donnée

## Table des matières

- 🪧 [À propos](#à-propos)
- 📦 [Prérequis](#prérequis)
- 🚀 [Installation](#installation)
- 🛠️ [Utilisation](#utilisation)
- 🤝 [Contribution](#contribution)
- 🏗️ [Construit avec](#construit-avec)
- 📚 [Documentation](#documentation)
- 🏷️ [Gestion des versions](#gestion-des-versions)
- 📝 [Licence](#licence)

## Prérequis

IL faut avoir d'installé sur la machine python, pip et PostgreSQL

## Installation

Dans le dossier du projet via un terminal installer avec la commande pip install les lib suivante :
pandas==3.0.3
matplotlib==3.11.1
seaborn==0.13.2
scikit-learn==1.9.0
shap==0.52.0
pytest==9.1.1
fastapi==0.139.2
uvicorn==0.51.0
python-multipart==0.0.32
httpx2==2.7.0
psycopg==3.3.4
psycopg-binary==3.3.4
SQLAlchemy==2.0.51
joblib==1.5.3
uuid==1.30
DateTime==6.0
pytest-cov==7.1.0

ou sinon via pip install -r requirements.txt

## Utilisation

### entrainer et enregsitrer le model de prévision
vous rendre sur le fichier model.py dans le dossier model et lancer Run Python File

### déployer le projet sur un serveur local :
python -m uvicorn model.main:app --reload  

### lancement des tests en local
python -m pytest -q

### lancement tests avec rapport de couverture
python -m pytest --cov=model --cov=db --cov-report=term-missing

### obtenir les lib et leurs version actuellemnt installé dans le projet
python -m pip freeze     

## Contribution

Developpeur : N.Ulrick

## Construit avec

### Langages & Frameworks

Le langage python, le gestionnaire Git et gitflow, test écrit avec pytest

### Outils

#### CI

le fichier local de conf CI se trouve dans le dossier workflows dans .github sous le nom ci.yml
le fichier est autmatiquement détecté par github

/!\ attention le repository local doit être lié à votre repository distant sur github dans le cas
contraire il faut les relier via cette commande :
git remote add origin git@github.com:Username/repository_name.git ( via ssh )
ou
git remote add origin https://github.com/Username/repository_name.git ( via http )

#### Déploiement

le fichier local de conf CD se trouve dans le dossier workflows dans .github sous le nom deploy_hf
pour avoir un CD fonctionnel il vous faudra :
- un compte hugging face
- HF_TOKEN
- un space dédier créer sur hugging face 

## Documentation

hugging face : https://huggingface.co/
github : https://github.com/
swagger : http://127.0.0.1:8000/docs#/default

## Gestion des versions

la gestion des version des versions et tag sont faite à la main via les commandes suivante :
git tag -a vX.X.X -m "Release vX.X.X"    
git push origin vX.X.X   

## Licence

Voir le fichier [LICENSE](./LICENSE.md) du dépôt.
