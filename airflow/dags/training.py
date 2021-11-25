from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'training',
    default_args=default_args,
    description='Training DAG',
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['training'],
) as dag:
    data_preparation = BashOperator(
        task_id="prepare_data",
        depends_on_past=False,
        bash_command="cd {{dag_run.conf['code_dir']}}; python3 prepare_data.py --data_path {{dag_run.conf['data_path']}} --verbose {{dag_run.conf['verbose']}}"
    )

    classifier_training = BashOperator(
        task_id="classifier_training",
        bash_command="cd {{dag_run.conf['code_dir']}}; python3 train.py --batch_size {{dag_run.conf['batch_size']}} --image_shape {{dag_run.conf['image_shape']}} --model_type {{dag_run.conf['model_type']}} --epochs {{dag_run.conf['epochs']}} --data_path {{dag_run.conf['data_path']}} --verbose {{dag_run.conf['verbose']}}",
    )

    classifier_evaluation = BashOperator(
        task_id="classifier_evaluation",
        bash_command="cd {{dag_run.conf['code_dir']}}; python3 eval.py --batch_size {{dag_run.conf['batch_size']}} --image_shape {{dag_run.conf['image_shape']}} --model_type {{dag_run.conf['model_type']}} --weights_path {{dag_run.conf['weights_path']}} --data_path {{dag_run.conf['test_data_path']}} --verbose {{dag_run.conf['verbose']}}",
    )

    drift_detector_training = BashOperator(
        task_id="drift_detector_training",
        bash_command="""cd {{dag_run.conf['code_dir']}}; python3 drift_training/train.py --data_path_pattern "{{dag_run.conf['drift_data_path_pattern']}}" --model_save_path {{dag_run.conf['drift_model_save_path']}}"""
    )

    model_mv = BashOperator(
        task_id="model_mv",
        bash_command="cd {{dag_run.conf['code_dir']}}; mkdir -p models/classifier; mv {{dag_run.conf['weights_path']}} models/classifier/model.h5"
    )

    deploy = BashOperator(
        task_id="deploy",
        bash_command="cd {{dag_run.conf['code_dir']}}; docker-compose up -d --build"
    )

    data_preparation >> [drift_detector_training, classifier_training] >> classifier_evaluation >> model_mv >> deploy
