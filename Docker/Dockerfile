FROM bitnami/spark:latest

# Install pip and Python libraries
USER root
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install jupyterlab pyspark hyperopt delta-spark

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
