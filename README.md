# 🌸 Iris Sepal Length Prediction Pipeline

## 📋 Overview
End-to-end ML pipeline predicting iris sepal length from sepal width using Docker, PostgreSQL, MLflow, and Flask/FastAPI. Built for botanical research automation.

## 🏗️ Architecture
```
┌─────────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
│Preprocessing├───►│PostgreSQL├───►│ MLflow  ├───►│API REST │
└─────────────┘    └──────────┘    └─────────┘    └─────────┘
```

## 📁 Project Structure
```
DATA_PIPELINE/
├── api/                    # REST API service
├── preprocessing/          # Data cleaning
├── mlflow/                # Model training
├── postgres/              # DB initialization
├── test/                  # Unit tests
├── docker-compose.yml     # Orchestration
├── requirements.txt
└── README.md
```

## 📡 API Usage
```bash
# POST /predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_width": 3.5}'

# Response
{"predicted_sepal_length": 5.84}
```

## 📊 Data

**File:** `iris.csv` (150 samples)  
**Columns:** `sepal_length` (target), `sepal_width` (feature), `petal_length`, `petal_width`, `species`

⚠️ **Note:** Document shows duplicate column names (`petal length` / `petal.length`) - verify actual CSV structure.



## 🛠️ Tech Stack

**Core:** Docker, Docker Compose, PostgreSQL 14+, MLflow 2.10+  
**API:** Flask/FastAPI  
**ML:** Scikit-learn (RandomForest), Pandas, SQLAlchemy

## 📈 Tracked Metrics

MLflow monitors: RMSE, MAE, R² Score

## 📚 Resources

[Docker](https://docs.docker.com/) • [MLflow](https://mlflow.org/docs/latest/) • [PostgreSQL](https://www.postgresql.org/docs/) • [Flask](https://flask.palletsprojects.com/) • [FastAPI](https://fastapi.tiangolo.com/) • [Scikit-learn](https://scikit-learn.org/stable/)


