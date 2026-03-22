# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a teaching repository for Apache Spark and ML with Databricks. It contains:
- **Demos/** — instructor-led notebooks with `<TODO>` placeholders for live coding
- **Labs/** — student exercise notebooks with `<TODO>` placeholders
- **Soluciones/** — completed reference solutions for all demos and labs
- **Docker/** — local Spark + JupyterLab environment for running notebooks outside Databricks

## Running locally (Docker)

```bash
cd Docker
docker-compose up
```

JupyterLab runs at `http://localhost:8888` (no token/password). Notebooks go in `Docker/notebooks/`.

The Docker image uses `bitnami/spark` and installs: `jupyterlab`, `pyspark`, `hyperopt`, `delta-spark`.

Local notebooks must create a `SparkSession` manually (unlike Databricks where `spark` is pre-injected):

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("...").master("local[*]").getOrCreate()
```

## Databricks notebooks

All files in `Demos/`, `Labs/`, and `Soluciones/` are Databricks notebooks (`.py` format with `# Databricks notebook source` header). They use:
- `# COMMAND ----------` to delimit cells
- `# MAGIC %md` for markdown cells
- `spark` and `dbutils` as pre-injected globals (no SparkSession creation needed)
- Delta Lake paths under `dbfs:/FileStore/` or Volumes

Data flows across notebooks sequentially (e.g., cleansed data from Demo 2 is consumed by Demo 3 onward). The Airbnb dataset cleaned in Demo 2 is stored at `dbfs:/FileStore/output/airbnb/clean_data` and reused by Demos 3–9.

## Notebook progression

1. Anonymization & Data Cleansing (PySpark SQL functions, deduplication)
2. Data Cleansing (SSN formatting, `initcap`, `sha2` anonymization)
3. Linear Regression (`VectorAssembler`, `Pipeline`, `RegressionEvaluator`)
4. Variable Discretization (`Bucketizer`, `QuantileDiscretizer`)
5. Decision Trees
6. Random Forests & Tuning (`CrossValidator`, `ParamGridBuilder`)
7. Hyperopt (`fmin`, `tpe`, `Trials`, search space with `hp.quniform`)
8. XGBoost
9. MLflow Tracking (`mlflow.start_run`, logging params/metrics/models/artifacts)
9.5. MLflow Model Registry

## Conventions for `<TODO>` placeholders

- `Demos/` files have `<TODO>` in key spots for live instructor coding
- `Soluciones/` files are the completed versions — always check these when filling in demos
- Labs have more extensive `<TODO>` blocks for student exercises
