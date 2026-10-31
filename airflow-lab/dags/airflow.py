# Import necessary libraries and modules
from airflow import DAG
from airflow.sdk import Param
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, find_optimal_k, train_final_model, predict_test

# Define default arguments for your DAG
default_args = {
    'owner': 'Nikhil Hugar',
    'start_date': datetime(2026, 1, 15),
    'retries': 2,  # a failed task is tried again up to 2 more times
    'retry_delay': timedelta(seconds=30),  # wait between retries
}

with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='K-Means pipeline that trains the final model with the k from the elbow method',
    catchup=False,
    # settings that can be changed in the trigger form
    params={
        "max_k": Param(49, type="integer", minimum=3, maximum=100,
                       description="Largest k to try in the elbow method"),
    },
    # so "{{ params.max_k }}" is passed to the task as a number, not text
    render_template_as_native_obj=True,
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # tries k = 1 to max_k and returns the best k from the elbow method
    find_optimal_k_task = PythonOperator(
        task_id='find_optimal_k_task',
        python_callable=find_optimal_k,
        op_args=[data_preprocessing_task.output],
        op_kwargs={"max_k": "{{ params.max_k }}"},
        execution_timeout=timedelta(minutes=10),
    )

    # trains the final model with the k from find_optimal_k_task (passed through XCom)
    train_final_model_task = PythonOperator(
        task_id='train_final_model_task',
        python_callable=train_final_model,
        op_args=[data_preprocessing_task.output, find_optimal_k_task.output, "model.sav"],
    )

    predict_task = PythonOperator(
        task_id='predict_task',
        python_callable=predict_test,
        op_args=["model.sav"],
    )

    load_data_task >> data_preprocessing_task >> find_optimal_k_task >> train_final_model_task >> predict_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()