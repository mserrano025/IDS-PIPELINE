
import argparse
import os
import subprocess

import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from scapy.all import sniff, wrpcap

class TrafficModelTrainer:
    def __init__(self, model_save_path="./models/"):
        """Starts the H2O server and sets the model save path."""
        self.model_save_path = model_save_path
        h2o.init(min_mem_size='10G')
    
    def load_and_prepare_data(self, file_path):
        """Loads and prepares the data from a CSV file."""
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', errors='replace', low_memory=False)
        
        df.columns = df.columns.str.strip()
        df['Label_Binary'] = df['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')
        
        # Convert 'Timestamp' to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp')
        
        # Force 'Label_Binary' to be categorical
        df['Label_Binary'] = df['Label_Binary'].astype('category')
        
        h2o_frame = h2o.H2OFrame(df)

        # Asign the 'Label_Binary' column as a target
        h2o_frame['Label_Binary'] = h2o_frame['Label_Binary'].asfactor()

        return h2o_frame

    def load_all_data(self, file_paths):
        """Loads and combines all the data from the given file paths."""
        training_frames = [self.load_and_prepare_data(file) for file in file_paths]
        combined_data = training_frames[0]
        for frame in training_frames[1:]:
            combined_data = combined_data.rbind(frame)
        return combined_data

    def train_binary_model(self, data, predictors, response_binary, max_runtime_secs=600):
        """Trains a binary classification model using H2O AutoML."""
        aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1234, verbosity="info", nfolds=5, keep_cross_validation_predictions=False)
        aml.train(x=predictors, y=response_binary, training_frame=data)
        return aml.leader

    def train_and_save_model(self, file_paths, predictors, response_binary, max_runtime_secs=600):
        """Loads, trains, and saves a binary classification model."""
        print("Loading data...")
        training_data = self.load_all_data(file_paths)
        print("Data loaded. Training model...")
        model = self.train_binary_model(training_data, predictors, response_binary, max_runtime_secs=max_runtime_secs)
        print("Model trained. Saving model...")
        model_path = h2o.save_model(model=model, path=self.model_save_path, force=True)
        print(f"Model loaded: {model_path}")
        return model_path

    def stop_h2o(self):
        """Shuts down the H2O server."""
        h2o.cluster().shutdown(prompt=False)
        print("H2O server stopped.")

# Class to analyze live traffic
class LiveTrafficAnalyzer:
    def __init__(self, model_path, pcaps_dir="./pcaps/", csv_dir="./csvs/", cicflowmeter_bin="./CICFlowMeter-4.0/bin"):
        self.model_path = model_path
        self.pcaps_dir = pcaps_dir
        self.csv_dir = csv_dir
        self.cicflowmeter_bin = cicflowmeter_bin

        # Create directories if they don't exist
        os.makedirs(self.pcaps_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        # Starts the H2O server and loads the model
        h2o.init(min_mem_size='8G')
        self.model = h2o.load_model(model_path)
        print(f"Model loaded: {model_path}")

    def capture_traffic(self, duration=60, pcap_file="live_traffic.pcap"):
        """Captures network traffic for a given duration and saves it to a pcap file."""
        pcap_path = os.path.join(self.pcaps_dir, pcap_file)
        print(f"Capturing traffic during {duration} seconds...")
        
        packets = sniff(timeout=duration)
        wrpcap(pcap_path, packets)
        
        print(f"Traffic captured and stored in {pcap_path}")
        return pcap_path

    def preprocess_pcap(self, pcap_path, csv_name="live_traffic.csv"):
        """Converts a pcap file to a CSV file using CICFlowMeter."""
        csv_path = os.path.join(self.csv_dir, csv_name)
        print(f"Proccessing {pcap_path} with CICFlowMeter to generate {csv_path}...")
        
        # Command to run CICFlowMeter
        command = f"sudo ./{self.cicflowmeter_bin}/cfm {pcap_path} {self.csv_dir}"
        subprocess.run(command, shell=True, check=True)
        csv_path = os.path.join(self.csv_dir, os.path.splitext(os.path.basename(pcap_path))[0] + ".pcap_Flow.csv")
        print(f"Preprocessing complete. Storing itv in {csv_path}")
        return csv_path

    def classify_csv(self, csv_path):
        """Classifies the data in a CSV file using the loaded model."""
        print(f"Clasifying data in {csv_path}...")
        
        # Load CSV file
        df = pd.read_csv(csv_path)

        # Convert to H2OFrame
        h2o_frame = H2OFrame(df)

        # Remove 'Label' column
        predictions = self.model.predict(h2o_frame)
        
        # Combine the predictions with the original data
        results = df.copy()
        results['Prediction'] = predictions.as_data_frame()['predict']

        print("Classification complete. Preview:")
        print(results[['Src IP', 'Dst IP', 'Prediction']])

        return results

    def analyze_live_traffic(self, duration=60):
        """All-in-one method to capture, preprocess, and classify live traffic."""
        # Capture traffic
        pcap_path = self.capture_traffic(duration=duration)

        # Preproccess the pcap
        csv_path = self.preprocess_pcap(pcap_path)

        # Classify the CSV
        results = self.classify_csv(csv_path)

        return results

def main():
    parser = argparse.ArgumentParser(description="Script for training a traffic classification model and analyzing live traffic.")
    parser.add_argument('-t', '--train', action='store_true', help="Train a traffic classification model.")
    parser.add_argument('-l', '--live', action='store_true', help="Capture and analyze live traffic.")
    args = parser.parse_args()

    # Route configuration
    model_save_path = "./models/"
    model_name = "GBM_1_AutoML_1_20241123_115607" # This model is already trained, has to be changed if the model changes
    model_path = os.path.join(model_save_path, model_name)

    if args.train:
        trainer = TrafficModelTrainer(model_save_path=model_save_path)
        trainer.train_and_save_model(
            file_paths=[
                "CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
                "CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv",
                "CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv",
                "CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                "CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                "CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
            ],
            predictors = [
                "Protocol","Flow Duration", "Total Fwd Packets", 
                "Total Backward Packets", "Total Length of Fwd Packets", 
                "Total Length of Bwd Packets", "Fwd Packet Length Max", 
                "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", 
                "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", 
                "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", 
                "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", 
                "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", 
                "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", 
                "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", 
                "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", 
                "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", 
                "Packet Length Mean", "Packet Length Std", "Packet Length Variance", 
                "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", 
                "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", 
                "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", 
                "Avg Bwd Segment Size", "Fwd Header Length", "Fwd Avg Bytes/Bulk", 
                "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", 
                "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", 
                "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", 
                "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", 
                "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", 
                "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
            ],
            response_binary = "Label_Binary",
            max_runtime_secs=600
        )
        trainer.stop_h2o()
    elif args.live:
        while True:
            try:
                analyzer = LiveTrafficAnalyzer(model_path=model_path)
                analyzer.analyze_live_traffic(duration=60)
            except KeyboardInterrupt:
                print("Saliendo...")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
    else:
        print("Please specify either --train or --live.")

if __name__ == "__main__":
    main()
