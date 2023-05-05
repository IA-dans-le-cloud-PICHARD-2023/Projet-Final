import glob
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.models import DAG
from datetime import datetime, timedelta
from textwrap import dedent
import pandas as pd
import seaborn as sns
import numpy as np

path = "/home/root123/Documents/cours/Projet-Final"


with DAG(
    "Spotifyhit-training",

    default_args={
        "depends_on_past": False,
        "email": ["spotihit@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "owner": "airflow",
    },
                                                                                         
    description="DAG to pull the data and train the model",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["spotifyhit"],
) as dag:


    # t1, t2 and t3 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="get_spotify_data",
        bash_command="source "+path+"/spotifyhit-env/bin/activate && cd "+path+" && python3 " + path + "/script-dag/get-data.py",
    )

    t2 = BashOperator(
        task_id="training",
        bash_command="source "+path+"/spotifyhit-env/bin/activate && cd "+path+" && python3 " + path + "/script-dag/training-xg.py",
        # bash_command="source "+path+"/spotifyhit-env/bin/activate",
    )

    t3 = BashOperator(
        task_id="api",
        bash_command="source "+path+"/spotifyhit-env/bin/activate && cd "+path+" && sudo -s docker build --tag spotifyhit-api . && sudo -s docker-compose up -d",
    )

    t1 >> t2 >> t3