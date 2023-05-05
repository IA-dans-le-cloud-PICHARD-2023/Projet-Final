Voici mon projet final pour le cours de IA dans le Cloud.

reauierements : Python > 3.10, pip, nginx (sudo apt-get install nginx), docker (sudo apt-get install docker), docker-compose (sudo apt-get install docker-compose)

le principe de ce projet et de pouvoir en quelques cliques recuperer des donnees, entrainer un modele, le superviser, et le deployer via une api 
qui va predire si une chanson est un hit ou non.

(definition de HIT : une chanson qui peux devenir mainstream et qui peux etre ecouter par un grand nombre de personne)

pour lancer le projet, lancer le script `start.py` qui install virtual env et airflow si il n'est pas sur la machine, ensuite il creer un venv,
install les bibliotheques requirements.txt dans le venv, et lance airflow en mode standalone.
une fois lancer, rendez vous sur `http://localhost:8080` pour voir le dags "spotifyhit-training".

vous pouvez le lancer manuellement ou attendre qu'il se lance automatiquement (il le fais une fois par jour donc mieux vaut le lancer manuellement).
ce dag va recuperer tout fichier csv contenue dans le dossier 'kaggle/input/datasets', les concatener et enregistrer le fichier final dans le dossier 'output/datas/combined_datasets.csv'.
ensuite il va entrainer un modele de XGBOOST sur ces donnees et l'enregistrer dans le dossier 'output/models'.

les performances du modele sont enregistrer dans mlflow pour les voir lancer la commande `mlflow ui` dans le dossier du projet et rendez vous sur `http://localhost:5000`.

une fois le modele entrainer, le dag va le deployer sur une api flask dans un contener docker load balancer par nginx, rendez vous sur `http://localhost:5001.
une interface web aparaitra et vous pourrez tester l'api en entrant le nom de la musique avec l'artiste.
dans le champ musique pas besoin de faire tres attention a l'orthographe et la maniere dont c'est ecrit. il faut simplement que se soit comprehensible et on ne puisse pas confondre avec d'autre musique.
exemple : "despacito, luis fonci", "luis fonci, despacito", "despacito", "despacito louis fonci" sont tous valide.


arboresence du projet :
--app
  |__app.py       # contient le code de l'api flask V2 qui devais faire du queuing (ToDo)
  |__templates    # contient le template html de l'interface web V2 qui devais faire du queuing (ToDo)
    |__indexV2.html

--dags
  |__job-spoti.py  # contient le dag qui va executer les codes python pour recuperer les donnees et entrainer le modele. il va aussi creer le contenair et le deployer a la suite

--kaggle
    |__input
        |__datasets   # contient les fichiers csv de kaggle ou autres qui von etre ensuite concatener

--outputs
    |__datas
        |__combined_datasets.csv  # contient le fichier csv final qui est le resultat de la concatenation des fichiers csv de kaggle
    |__models
        |__Final_model_xgb.bin  # contient le modele entrainer

--script-dag
    |__get-data.py  # contient le code python qui va recuperer les donnees et les concatener
    |__training-xg.py  # contient le code python qui va entrainer le modele
-
-spotifyhit-env  # contient le venv cree par le start.py

--templates
    |__index.html  # contient le template html de l'interface web V1\

--app.py # contient le code de l'api flask V1

--docker-compose.yml  # contient la configuration du contenair docker

--Dockerfile  # contient la configuration du contenair docker

--start.py
