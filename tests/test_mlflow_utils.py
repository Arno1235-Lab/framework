import os
import pytest
import tempfile
import mlflow
import numpy as np
import sklearn.datasets
import sklearn.linear_model
from sklearn.model_selection import train_test_split

import sys
sys.path.append('./..')

# Import the MLflow utils we created earlier
from mlflow_utils import (
    save_training_metrics,
    register_model_with_alias,
    load_model_by_alias,
    is_mlflow_server_running,
    start_nested_run,
    setup_mlflow_tracking
)

@pytest.fixture(scope="session")
def mlflow_tracking():
    """
    Fixture to setup and teardown MLflow tracking for tests
    """
    # Create a temporary tracking directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set tracking to local directory
        mlflow.set_tracking_uri(f"file:{tmp_dir}")
        # mlflow.set_tracking_uri("file:mlruns")

        # Set a test experiment
        mlflow.set_experiment("test_mlflow_utils")
        
        yield tmp_dir

@pytest.fixture
def sample_model_and_data():
    """
    Create a sample sklearn model and dataset for testing
    """
    # Load iris dataset
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train a simple model
    model = sklearn.linear_model.LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def test_save_training_metrics(mlflow_tracking):
    """
    Test saving nested training metrics
    """
    # Start an MLflow run
    with mlflow.start_run():
        # Prepare nested metrics
        metrics = {
            'train': {
                'loss': 0.123,
                'accuracy': 0.95,
                'details': {
                    'epochs': 10,
                    'batch_size': 32
                }
            },
            'eval': {
                'precision': 0.88,
                'recall': 0.90
            }
        }
        
        # Save metrics
        save_training_metrics(metrics)
        
        # Retrieve logged metrics
        run = mlflow.active_run()
        client = mlflow.tracking.MlflowClient()
        
        # Check some key metrics were logged
        logged_metrics = client.get_metric_history(run.info.run_id, "train/loss")
        assert len(logged_metrics) > 0
        assert logged_metrics[0].value == 0.123

def test_model_registration_and_loading(mlflow_tracking, sample_model_and_data):
    """
    Test model registration with alias and subsequent loading
    """
    model, X_test, _ = sample_model_and_data
    
    # Start a run and log the model
    with mlflow.start_run():
        # Log sklearn model with signature
        signature = mlflow.models.signature.infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature,
            input_example=X_test[0:1]
        )
        
        # Register model with alias
        model_version = register_model_with_alias(
            model_uri=mlflow.get_artifact_uri("model"),
            model_name="test_model",
            alias="best_model"
        )
        # Check if the correct version is returned
        assert model_version == 1

        # Register new version of model with alias
        model_version = register_model_with_alias(
            model_uri=mlflow.get_artifact_uri("model"),
            model_name="test_model",
            alias="best_model"
        )
        # Check if the correct version is returned
        assert model_version == 2
        
        # Load model by alias
        loaded_model = load_model_by_alias(
            model_name="test_model", 
            alias="best_model",
            flavor="sklearn"
        )
        
        # Verify model works
        assert hasattr(loaded_model, "predict")

def test_nested_run(mlflow_tracking):
    """
    Test starting nested runs
    """
    # Start a parent run
    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
        
        # Start a nested run
        with start_nested_run(parent_run_id=parent_run_id, nested_run_name="fine_tuning"):
            mlflow.log_metric("fine_tune_metric", 0.99)
        
        # Verify nested run was logged
        client = mlflow.tracking.MlflowClient()
        nested_runs = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name("test_mlflow_utils").experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        )
        
        assert len(nested_runs) >= 1
        
        # Check if the nested run's metric was logged
        if nested_runs:
            assert nested_runs[0].data.metrics.get("fine_tune_metric") == 0.99

def test_mlflow_server_running():
    """
    Test MLflow server running check
    
    Note: This test might need adjustment based on your local setup
    """
    # Check local default MLflow server (which may not be running)
    result = is_mlflow_server_running()
    
    # This could be True or False depending on local environment
    # The important thing is that it doesn't raise an exception
    assert isinstance(result, bool)

def test_setup_mlflow_tracking(mlflow_tracking):
    """
    Test MLflow tracking setup
    """
    # Create a new temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"file:{tmp_dir}"
        # tracking_uri = "file:mlruns"
        experiment_name = "test_setup_tracking"
        
        # Setup tracking
        setup_mlflow_tracking(
            tracking_uri=tracking_uri, 
            experiment_name=experiment_name
        )
        
        # Verify tracking URI
        assert mlflow.get_tracking_uri() == tracking_uri
        
        # Verify experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None

# Optional: main block for direct script execution
if __name__ == '__main__':
    pytest.main([__file__])