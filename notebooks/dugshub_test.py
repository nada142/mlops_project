import mlflow 
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/nada142/mlops_project.mlflow")

dagshub.init(repo_owner='nada142', repo_name='mlops_project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)