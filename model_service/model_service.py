import requests
import h2o
import pandas as pd
import time
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

h2o.init(min_mem_size='8G')

# Configuration
DATA_SERVICE_URL = "http://data-service:5000/get_data"
DATA_SERVICE_START_URL = "http://data-service:5000/start"
THRESHOLD_PRECISION = 0.6
THRESHOLD_FALSE_NEGATIVES = 100
METRICS_LOG_PATH = "logs/model_metrics_log_Thursday_k8s.csv"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Send request to data-service to start streaming
try:
    response = requests.post(DATA_SERVICE_START_URL)
    if response.status_code == 200:
        print("Data-service started.")
    else:
        print(f"Error in data-service: {response.status_code}")
except Exception as e:
    print(f"Can`t connect to data-service: {e}")

# Load model
try:
    os.system("ls models/")
    model = h2o.load_model("models/final_model_Wednesday")
except:
    print("Model not found. Training new model...")
    df = pd.read_csv("CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    h2o_df = h2o.H2OFrame(df)
    model = H2OAutoML(max_runtime_secs=60).train(x=h2o_df.columns[:-1], y="Label_Binary", training_frame=h2o_df)
    h2o.save_model(model=model, path="models/final_model_Thursday", force=True)

# Create metrics log file
with open(METRICS_LOG_PATH, "w") as f:
    f.write("timestamp,algorithm,accuracy,f1,recall,precision,true_positives,true_negatives,false_positives,false_negatives,retrained\n")

while True:
    response = requests.get(DATA_SERVICE_URL)

    if response.status_code == 404:
        print("No more data. Waiting 10 seconds...")
        time.sleep(10)
        continue

    data = pd.DataFrame(response.json())
    h2o_data = h2o.H2OFrame(data)

    predictions = model.predict(h2o_data)['predict'].as_data_frame()
    accuracy = accuracy_score(data['Label_Binary'], predictions)
    precision = precision_score(data['Label_Binary'], predictions, pos_label='ATTACK')
    recall = recall_score(data['Label_Binary'], predictions, pos_label='ATTACK')
    f1 = f1_score(data['Label_Binary'], predictions, pos_label='ATTACK')
    confusion = confusion_matrix(data['Label_Binary'], predictions, labels=['BENIGN', 'ATTACK'])

    true_positives = confusion[0][0]
    true_negatives = confusion[1][1]
    false_positives = confusion[0][1]
    false_negatives = confusion[1][0]

    retrained = False

    if precision < THRESHOLD_PRECISION and false_negatives > THRESHOLD_FALSE_NEGATIVES:
        print("Reentrenando modelo...")
        new_data = h2o.H2OFrame(data)
        model = H2OAutoML(max_runtime_secs=60).train(x=new_data.columns[:-1], y="Label_Binary", training_frame=new_data)
        h2o.save_model(model=model, path="models/final_model_Thursday", force=True)
        retrained = True

    # Save metrics to log file
    with open(METRICS_LOG_PATH, "a") as f:
        f.write(f"{time.strptime},gbm,{accuracy},{f1},{recall},{precision},{true_positives},{true_negatives},{false_positives},{false_negatives},{retrained}\n")

    print(f"Registered metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    time.sleep(5)
