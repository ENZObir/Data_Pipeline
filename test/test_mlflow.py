#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import mlflow

def main() -> None:
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Iris_Project")

    with mlflow.start_run(run_name="quick-check"):
        mlflow.log_metric("accuracy", 0.95)
        print(f"Logged accuracy metric to {tracking_uri}")

if __name__ == "__main__":
    main()