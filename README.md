# 🌸 Iris Sepal Length Prediction Pipeline

## 📋 Overview
End-to-end ML pipeline predicting iris sepal length from sepal width using Docker, PostgreSQL, MLflow, and Flask/FastAPI. Built for botanical research automation.

## 🏗️ Architecture
```
┌─────────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐
│Preprocessing├───►│PostgreSQL├───►│ MLflow  ├───►│API REST │
│   Service   │    │          │    │Training │    │ Service │
└─────────────┘    └──────────┘    └─────────┘    └─────────┘
│                                  │              │
▼                                  ▼              ▼
CSV Exports                      Model Registry   Predictions
(clean + 3 species)             (4 trained models)
│                                  │
└──────────────┬───────────────────┘
▼
┌──────────────┐
│  MLflow UI   │
│   Service    │
└──────────────┘
```

**5 Dockerized Services:**

1. **Preprocessing Service**
   - Loads raw `iris.csv`
   - Cleans and validates data
   - Exports 4 CSV files (clean + 3 species chunks)
   - Inserts data into PostgreSQL (4 tables)
   - **Status:** Exits after completion

2. **PostgreSQL Service**
   - Stores cleaned data in 4 separate tables
   - Provides persistent data storage
   - **Status:** Always running

3. **MLflow Training Service**
   - Extracts data from PostgreSQL
   - Trains 4 RandomForestRegressor models:
     - 1 general model (all species)
     - 3 species-specific models (Setosa, Versicolor, Virginica)
   - Logs experiments and metrics to MLflow
   - Registers models in MLflow Model Registry
   - **Status:** Exits after training completion

4. **MLflow UI Service**
   - Web interface for experiment tracking
   - Model comparison and metrics visualization
   - **Status:** Always running

5. **API REST Service**
   - Loads best models from MLflow Registry
   - Exposes prediction endpoints (general + species-specific)
   - Serves predictions via `/predict` routes
   - **Status:** Always running

## 🔄 Automated Workflow


Container orchestration with `depends_on` ensures sequential execution:

**Step 1: Preprocessing** (runs first, exits on completion)
- Loads `api/iris.csv`
- Cleans and validates data
- Exports 4 CSV files to `data/processed/`:
  - `iris_clean.csv` (all cleaned data, ~150 rows)
  - `iris_setosa.csv` (Setosa species only, ~50 rows)
  - `iris_versicolor.csv` (Versicolor species only, ~50 rows)
  - `iris_virginica.csv` (Virginica species only, ~50 rows)
- Inserts data into PostgreSQL (4 tables)

**Step 2: ML Training** (waits for preprocessing, exits on completion)
- Queries data from PostgreSQL tables
- Trains 4 RandomForestRegressor models
- Logs metrics (RMSE, MAE, R²) to MLflow
- Registers models in MLflow Registry

**Step 3: Services Deployment** (waits for ML training, always running)
- MLflow UI becomes accessible
- API loads models and serves predictions

## 📊 Data

**Source:** `iris.csv` (150 observations)

**Columns:**
- `sepal_length` : Sepal length (target variable)
- `sepal_width` : Sepal width (predictive feature)
- `petal_length` : Petal length
- `petal_width` : Petal width
- `species` : Flower species (Setosa, Versicolor, Virginica)

## 📂 Data Flow & Chunking Strategy
```
api/iris.csv (raw 150 rows)
       ↓
preprocessing service
       ↓
data/processed/
  ├── iris_clean.csv        # All species cleaned (150 rows)
  ├── iris_setosa.csv       # Setosa only (~50 rows)
  ├── iris_versicolor.csv   # Versicolor only (~50 rows)
  └── iris_virginica.csv    # Virginica only (~50 rows)
       ↓
PostgreSQL tables:
  ├── iris_clean
  ├── iris_setosa
  ├── iris_versicolor
  └── iris_virginica
       ↓
MLflow Training Service
       ↓
MLflow Model Registry:
  ├── iris_general_model      # Trained on all data
  ├── iris_setosa_model       # Specialized for Setosa
  ├── iris_versicolor_model   # Specialized for Versicolor
  └── iris_virginica_model    # Specialized for Virginica
    
Why chunk by species?
- Different species may have different sepal length/width relationships
- Enables species-specific model comparison
- Potentially better model performance for known species
- Demonstrates data partitioning and specialized modeling skills
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





## 🛠️ Tech Stack

**Core:** Docker, Docker Compose, PostgreSQL 14+, MLflow 2.10+  
**API:** Flask/FastAPI  
**ML:** Scikit-learn (RandomForest), Pandas, SQLAlchemy

## 📈 Tracked Metrics

MLflow monitors: RMSE, MAE, R² Score

## 📚 Resources

[Docker](https://docs.docker.com/) • [MLflow](https://mlflow.org/docs/latest/) • [PostgreSQL](https://www.postgresql.org/docs/) • [Flask](https://flask.palletsprojects.com/) • [FastAPI](https://fastapi.tiangolo.com/) • [Scikit-learn](https://scikit-learn.org/stable/)

## 🛠️ Troubleshooting

- **Check container status**: `docker compose ps` verifies health/exit codes before digging into logs.
- **Inspect preprocessing/training logs**: `docker compose logs preprocessing training --tail 100` confirms data landed in PostgreSQL before modeling starts.
- **Validate database readiness**: `docker compose exec db pg_isready -U $POSTGRES_USER -d $POSTGRES_DB` mirrors the healthcheck used by the orchestrator.
- **MLflow diagnostics**: `docker compose logs mlflow --tail 100` shows registry activity and artifact storage paths if the API cannot load a model.
- **API errors**: `docker compose logs api --tail 100` surfaces FastAPI stack traces (e.g., missing model URI, validation issues).
- **Persistent volumes**: `docker volume inspect datapipeline-docker-public_pgadmin-data` (replace with the volume name) confirms that PostgreSQL and MLflow data survive `docker compose down`.


