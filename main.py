import mlflow
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        steps_to_execute = list(config["main"]["execute_steps"])


    # Download step
    if "download" in steps_to_execute:
        logger.info("Downloading and reading artifact")

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:
        logger.info("Preprocessing data")

        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "preprocessed_data.csv",
                "artifact_type": "processed_data",
                "artifact_description": "Data after processing",
            },
        )
        pass

    if "check_data" in steps_to_execute:
        logger.info("Checking preprocessed data")

        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": config["data"]["reference_dataset"],
                "sample_artifact":"preprocessed_data.csv:latest",
                "ks_alpha": config["data"]["ks_alpha"],
            },
        )        
        pass

    if "segregate" in steps_to_execute:
        logger.info("Stratifying data")
  
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact":"preprocessed_data.csv:latest",
                "artifact_root":"dataset",
                "artifact_type":"stratified_data",
                "test_size":config["data"]["test_size"],
                "stratify": config["data"]["stratify"],                
            },       
        )
        pass

    if "random_forest" in steps_to_execute:
        logger.info("Random forest")

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))


        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data":"dataset_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"],    
            },
        )
        pass

    if "evaluate" in steps_to_execute:
        logger.info("Evaluating model")

        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data":"dataset_test.csv:latest",                
            }
        )
        pass


if __name__ == "__main__":
    go()
