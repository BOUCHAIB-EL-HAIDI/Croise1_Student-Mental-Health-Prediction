
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.data.data_profiling import main as profile_data
from src.training.model_comparison import main as compare_models
from src.training.hyperparameter_optimization import main as optimize_models
from src.evaluation.model_evaluation import main as evaluate_model


default_args = {
    "owner": "student",
    "retries": 2,
    "retry_delay": timedelta(minutes=6)
}


with DAG(
    dag_id="student_mental_health_pipeline",
    default_args=default_args,
    description="Student Mental Health Risk Prediction workflow",
    start_date=datetime(2026, 11, 13),
    schedule="@daily",
    catchup=False,
    tags=["machine-learning", "mental-health"]
) as dag:

    profiling_task = PythonOperator(
        task_id="data_profiling",
        python_callable=profile_data
    )

    comparison_task = PythonOperator(
        task_id="model_comparison",
        python_callable=compare_models
    )

    optimization_task = PythonOperator(
        task_id="hyperparameter_optimization",
        python_callable=optimize_models
    )

    evaluation_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_model
    )

    profiling_task >> comparison_task >> optimization_task >> evaluation_task
