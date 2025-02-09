# IDS-Framework

This repository provides an Intrusion Detection System (IDS) framework that can be executed in three different modes: locally, sniffing real network traffic, or in a distributed Kubernetes environment.

## Project Structure

The repository is organized as follows:

```bash
IDS-FRAMEWORK/ 
│── CICFlowMeter-4.0/ # Tool for network flow feature extraction 
│── CICIDS2017/ # Dataset used for training and evaluation 
│── csvs/ # Preprocessed CSV files 
│── data_service/ # Service for handling data processing in Kubernetes 
│── images/ # Contains different images: 
│ ├── attacks/ # Visualizations of detected attacks 
│ ├── console logs/ # Logs from executions 
│ └── metrics/ # Performance metrics visualizations 
│── k8s/ # Kubernetes deployment configurations 
│── logs/ # Execution logs 
│── metrics_csvs/ # CSV logs containing model performance metrics 
│── model_service/ # Service handling model inference in Kubernetes 
│── models/ # Trained model artifacts 
│── pcaps/ # Captured network traffic files 
│── .gitignore # Git ignore rules 
│── comparison_plotting_k8s.py # Script for comparing local vs Kubernetes performance 
│── h2o_framework_network_data.py # Pipeline for real-time traffic classification 
│── h2o_framework.py # Main IDS pipeline for local execution 
│── h2o.jar # H2O.ai framework binary 
│── minikube-linux-amd64 # Minikube binary for running Kubernetes locally 
│── README.md # This file 
│── requirements.txt # Python dependencies
```

## Running the IDS Framework

The IDS can be executed in three different modes:

### 1. Running the Pipeline Locally
To execute the IDS locally on preprocessed data:
```bash
python3 h2o_framework.py
```
This script loads the CICIDS2017 dataset (or other provided CSVs), preprocesses the data, and applies an AutoML model for classification.

### 2. Running the Pipeline with Real Network Traffic
To capture and classify live network traffic:
```bash
python3 h2o_framework_network_data.py
```
This script captures network traffic using the `scapy` library, preprocesses the data, and applies an AutoML model for classification.

### 3. Running the Pipeline in a Kubernetes Cluster
To deploy the IDS in a Kubernetes cluster:
```bash
eval $(minikube docker-env)
docker build -t model-service-image -f model_service/Dockerfile . && docker build -t data-service-image -f data_service/Dockerfile .
kubectl apply -f volumes/persistent-volume.yaml && kubectl apply -f volumes/persistent-volume-claim.yaml
kubectl apply -f services/model-service.yaml && kubectl apply -f services/data-service.yaml
kubectl apply -f deployments/data-service-deployment.yaml && kubectl apply -f deployments/model-service-deployment.yaml
```
This setup deploys:

- A data-service for handling network traffic preprocessing.
- A model-service that classifies network traffic in real-time.
- Persistent storage for managing logs and preprocessed traffic data.

## Dependencies
Before running the framework, install required dependencies:
```bash
pip3 install -r requirements.txt
```
If you plan to run the system in a Kubernetes environment, make sure you have Minikube and Docker installed and running.


