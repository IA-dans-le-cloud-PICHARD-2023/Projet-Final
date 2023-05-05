import os
import time


print("\n\n=====================================================")
print("creation du virtualenv\n")
os.system('pip install virtualenv')
os.system('pip install apache-airflow')
os.system('python3 -m venv spotifyhit-env')
# wait for the venv to be created
while not os.path.exists('spotifyhit-env/bin/activate'):
    time.sleep(4)
    pass


path = os.getcwd()
print("path = " + path)
time.sleep(5)


print("\n\n=====================================================")
print("telechargement des dependences\n")
# lancement et attente du debdown.sh
os.system('chmod +x ./debdown.sh')
os.system('./debdown.sh')

time.sleep(20)

print("\n\n=====================================================")
print("Copie du dags dans le dossier airflow/dags")

# copie du fichier job-spoti.py puis modification de la variable path
os.system('sed -i "s|path = .*|path = \\"'+path+'\\"|g" ./dags/job-spoti.py')



time.sleep(1)

os.system('deactivate')
os.system('export AIRFLOW_HOME=~/airflow')
os.system('cp ./dags/* ~/airflow/dags')
print("\n\n=====================================================")
print("lancement airflow\n")

os.system('airflow db init')
os.system('airflow standalone')
