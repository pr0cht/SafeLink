import time
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = 'safelink_model.tflite'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64
TEST_DATA_PATH = 'sms_datasets/test_data.csv' # We will create this below if it doesn't exist

# ==========================================
# 2. LOAD TFLITE MODEL
# ==========================================
print(f"Loading {MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper to find index by name (because TFLite reorders inputs)
input_map = {
    'input_ids': input_details[0]['index'],   # Adjust indices dynamically based on your model
    'attention_mask': input_details[1]['index'], 
    'url_features': input_details[2]['index']
}

# NOTE: You might need to swap indices [0], [1], [2] based on your specific build.
# We will print them to be sure.
print("\nModel Input Details:")
for i, detail in enumerate(input_details):
    print(f"Index {i}: {detail['name']} | Shape: {detail['shape']}")
    if 'input_ids' in detail['name']: input_map['input_ids'] = i
    elif 'attention_mask' in detail['name']: input_map['attention_mask'] = i
    elif 'url_features' in detail['name']: input_map['url_features'] = i

# ==========================================
# 3. PREPARE DUMMY TEST DATA (OR LOAD REAL)
# ==========================================
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Let's create a small "adversarial" test set manually for the benchmark
test_sentences = [
    "Your account is locked. Click http://bit.ly/scam now!",  # Phishing
    "Hey mom, I'll be home for dinner.",                      # Safe
    "URGENT: You won $5000! Claim at www.prize.com",          # Phishing
    "Meeting rescheduled to 3 PM.",                           # Safe
    "Verify your identity immediately: http://secure-bank.com" # Phishing
]
test_labels = [1, 0, 1, 0, 1] # 1=Phishing, 0=Safe

def extract_url_features(text):
    has_url = 1 if 'http' in text else 0
    len_url = len([w for w in text.split() if 'http' in w])
    is_shortened = 1 if 'bit.ly' in text or 'tinyurl' in text else 0
    return np.array([has_url, len_url, is_shortened], dtype=np.float32)

# ==========================================
# 4. BENCHMARK LOOP
# ==========================================
print("\nStarting Desktop Inference Benchmark...")
latencies = []
predictions = []

for i, text in enumerate(test_sentences):
    # --- Preprocessing ---
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='np' # Return Numpy directly for TFLite
    )
    
    input_ids = tokens['input_ids'].astype(np.int32)
    attention_mask = tokens['attention_mask'].astype(np.int32)
    url_feats = extract_url_features(text).reshape(1, 3)

    # --- Set Inputs ---
    # Use the mapped indices we found earlier
    interpreter.set_tensor(input_details[input_map['input_ids']]['index'], input_ids)
    interpreter.set_tensor(input_details[input_map['attention_mask']]['index'], attention_mask)
    interpreter.set_tensor(input_details[input_map['url_features']]['index'], url_feats)

    # --- INFERENCE (TIMED) ---
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    
    # --- Get Output ---
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = output_data[0][0]
    prediction = 1 if prob > 0.5 else 0
    
    # Log results
    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)
    predictions.append(prediction)
    
    print(f"Msg: '{text[:20]}...' | Prob: {prob:.4f} | Pred: {prediction} | Time: {latency_ms:.2f}ms")

# ==========================================
# 5. REPORT GENERATION
# ==========================================
avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)

print("\n" + "="*40)
print("DESKTOP BENCHMARK RESULTS (x86 Baseline)")
print("="*40)
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"p95 Latency:     {p95_latency:.2f} ms")
print("-" * 40)
print("Classification Report:")
print(classification_report(test_labels, predictions, target_names=['Safe', 'Phishing']))