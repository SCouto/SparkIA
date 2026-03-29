[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SCouto/SparkIA)

# SparkIA

Repository containing Demos and Labs for Apache Spark and ML with Databricks.

## Folder Structure

- **Demos/** — instructor-led notebooks with `<TODO>` placeholders for live coding
- **Labs/** — student exercise notebooks with `<TODO>` placeholders
- **Soluciones/** — completed reference solutions for all demos and labs
- **Datasets/** — data files used across notebooks
- **Docker/** — local Spark + JupyterLab environment

## Notebook Progression

Notebooks are designed to run sequentially. **Demo 2 (Data Cleansing)** generates the cleaned Airbnb dataset that all subsequent notebooks (3–9) depend on.

The output of **Demo 2 (Data Cleansing)** , and input for the following notebooks is already available in `Datasets/outpus/airbnb/` so you can jump straight to any later notebook.

## Running locally with Docker

```bash
cd Docker
docker-compose up --build
```

JupyterLab will be available at `http://localhost:8888` (no token or password required).

The following folders are mounted into the container and available under `/home/jovyan/work/`:

| Local folder  | Container path    |
| ------------- | ----------------- |
| `Demos/`      | `work/demos/`     |
| `Labs/`       | `work/labs/`      |
| `Soluciones/` | `work/solutions/` |
| `Datasets/`   | `work/datasets/`  |
