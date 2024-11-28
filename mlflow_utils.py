import os
import mlflow
import requests
from typing import Dict, Any, Optional

def save_training_metrics(metrics: Dict[str, Any]) -> None:
    """
    Save nested dictionary of training metrics to MLflow.
    
    Args:
        metrics (Dict[str, Any]): Nested dictionary of metrics to log.
                                  Supports multiple levels of nesting.
    
    Example:
        metrics = {
            'train': {
                'loss': 0.123,
                'accuracy': 0.95
            },
            'eval': {
                'precision': 0.88,
                'recall': 0.90
            }
        }
    """
    def _log_nested_metrics(prefix: str, metrics_dict: Dict[str, Any]):
        """Recursively log nested metrics."""
        for key, value in metrics_dict.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                _log_nested_metrics(full_key, value)
            elif isinstance(value, (int, float)):
                # Log numeric values
                mlflow.log_metric(full_key, value)
            else:
                # Log non-numeric values as params
                mlflow.log_param(full_key, str(value))
    
    _log_nested_metrics("", metrics)

def register_model_with_alias(
    model_uri: str, 
    model_name: str, 
    alias: str, 
) -> str:
    """
    Register a model with a specific alias.
    
    Args:
        model_uri (str): URI of the model to register (e.g., 'runs:/...')
        model_name (str): Name of the model in MLflow registry
        alias (str): Alias to assign to the model version
    
    Returns:
        str: Model version
    """
    client = mlflow.tracking.MlflowClient()
    
    # First, create the registered model if it doesn't exist
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        # If model doesn't exist, create it
        client.create_registered_model(model_name)
    
    # Register the model version
    model_version = client.create_model_version(
        name=model_name, 
        source=model_uri
    )
    
    # Add alias to the model version
    client.set_registered_model_alias(
        name=model_name, 
        alias=alias, 
        version=model_version.version
    )
    
    return model_version.version

def load_model_by_alias(
    model_name: str, 
    alias: str, 
    flavor: str = 'python_model'
) -> Any:
    """
    Load a model from MLflow registry using its alias.
    
    Args:
        model_name (str): Name of the model in MLflow registry
        alias (str): Alias of the model version to load
        flavor (str): MLflow model flavor to load (default: 'python_model')
    
    Returns:
        Loaded model object
    """
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(
        name=model_name, 
        alias=alias
    )
    
    # Load the model based on specified flavor
    if flavor == 'python_model':
        return mlflow.pyfunc.load_model(
            f"models:/{model_name}/{model_version.version}"
        )
    elif flavor == 'sklearn':
        return mlflow.sklearn.load_model(
            f"models:/{model_name}/{model_version.version}"
        )
    elif flavor == 'pytorch':
        return mlflow.pytorch.load_model(
            f"models:/{model_name}/{model_version.version}"
        )
    else:
        raise ValueError(f"Unsupported model flavor: {flavor}")

def is_mlflow_server_running(
    tracking_uri: str = 'http://localhost:5000'
) -> bool:
    """
    Check if the MLflow tracking server is running.
    
    Args:
        tracking_uri (str): URI of the MLflow tracking server 
                            (default: 'http://localhost:5000')
    
    Returns:
        bool: True if server is running, False otherwise
    """
    try:
        response = requests.get(
            f"{tracking_uri}/", 
            timeout=5
        )
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False

def start_nested_run(
    parent_run_id: Optional[str] = None, 
    nested_run_name: Optional[str] = None
) -> mlflow.ActiveRun:
    """
    Start a nested MLflow run, optionally under an existing run.
    
    Args:
        parent_run_id (Optional[str]): ID of the parent run to nest under
        nested_run_name (Optional[str]): Name for the nested run
    
    Returns:
        mlflow.ActiveRun: The started nested run
    """
    # If no parent run is active and no parent_run_id is provided, 
    # start a new top-level run
    if not mlflow.active_run() and not parent_run_id:
        mlflow.start_run()
    
    # Start the nested run
    nested_run = mlflow.start_run(
        run_name=nested_run_name, 
        nested=True
    )
    
    return nested_run

# Utility to set tracking URI and ensure it's configured
def setup_mlflow_tracking(
    tracking_uri: Optional[str] = None, 
    experiment_name: Optional[str] = None
):
    """
    Configure MLflow tracking URI and experiment.
    
    Args:
        tracking_uri (Optional[str]): MLflow tracking server URI
        experiment_name (Optional[str]): Name of the experiment to set
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment if provided
    if experiment_name:
        mlflow.set_experiment(experiment_name)
