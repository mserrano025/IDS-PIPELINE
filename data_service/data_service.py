from flask import Flask, jsonify, request
import pandas as pd

def load_and_prepare_data(file_path):
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
    
    return df

app = Flask(__name__)

# Load datasets
day_files = ["CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
             "CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"]

# Load and prepare data
selected_day_frames = [load_and_prepare_data(file) for file in day_files]
selected_day_data = pd.concat(selected_day_frames, ignore_index=True)

start_time = selected_day_data['Timestamp'].min()
end_time = selected_day_data['Timestamp'].max()
time_ranges = pd.date_range(start=start_time, end=end_time, freq='7min30s')

day_hours = [selected_day_data[(selected_day_data['Timestamp'] >= time_ranges[i]) & (selected_day_data['Timestamp'] < time_ranges[i+1])] 
             for i in range(len(time_ranges)-1)]
day_hours = [hour for hour in day_hours if len(hour) > 0]

segment_index = 0
is_running = False

def reset_index():
    global segment_index
    segment_index = 0

@app.route('/start', methods=['POST'])
def start_stream():
    global is_running
    is_running = True
    reset_index()
    return jsonify({"message": "Streaming started"}), 200

@app.route('/stop', methods=['POST'])
def stop_stream():
    global is_running
    is_running = False
    return jsonify({"message": "Streaming stopped"}), 200

@app.route('/get_data', methods=['GET'])
def get_data():
    global segment_index, is_running
    print(f"Request received. is_running: {is_running}, segment_index: {segment_index}, total_segments: {len(day_hours)}")
    
    if not is_running or segment_index >= len(day_hours):
        print("No more data. Returning 404.")
        return jsonify({"message": "No more data avaliable"}), 404

    data = day_hours[segment_index].to_dict(orient='records')
    segment_index += 1
    return jsonify(data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
