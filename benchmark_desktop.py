import os
import time
import psutil
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
MODEL_PATH = 'safelink_model.tflite'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64
NUM_ITERATIONS = 100  # Number of inferences to run per test

# A standard smishing message to process repeatedly 
# (We use the same text so the computational math is identical every time)
TEST_TEXT = "CHASE BANK: We detected an unauthorized login attempt. Secure your account at https://bit.ly/chase-auth-99"

print("Loading model and tokenizer (This is NOT timed)...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Force TFLite to use 1 thread so we can strictly measure single-core affinity
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=1)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map inputs dynamically
input_map = {}
for i, detail in enumerate(input_details):
    if 'input_ids' in detail['name']: input_map['input_ids'] = i
    elif 'attention_mask' in detail['name']: input_map['attention_mask'] = i
    elif 'url_features' in detail['name']: input_map['url_features'] = i

def extract_url_features(text):
    has_url = 1 if 'http' in text else 0
    len_url = len([w for w in text.split() if 'http' in w])
    is_shortened = 1 if 'bit.ly' in text else 0
    return np.array([has_url, len_url, is_shortened], dtype=np.float32)

# Pre-process the inputs once (we are measuring inference speed, not tokenizer speed)
tokens = tokenizer(TEST_TEXT, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='np')
input_ids = tokens['input_ids'].astype(np.int32)
attention_mask = tokens['attention_mask'].astype(np.int32)
url_feats = extract_url_features(TEST_TEXT).reshape(1, 3)

# ==========================================
# 2. THE BENCHMARK FUNCTION
# ==========================================
# ==========================================
# 2. THE BENCHMARK FUNCTION (FIXED)
# ==========================================
def run_benchmark(mode_name, pinned_cores):
    process = psutil.Process()
    
    # Apply OS Processor Affinity
    process.cpu_affinity(pinned_cores)
    print(f"\n--- Starting {mode_name} ---")
    print(f"Process locked to CPU Cores: {process.cpu_affinity()}")
    
    # --- FIX: Set the tensors BEFORE the warm-up run ---
    interpreter.set_tensor(input_details[input_map['input_ids']]['index'], input_ids)
    interpreter.set_tensor(input_details[input_map['attention_mask']]['index'], attention_mask)
    interpreter.set_tensor(input_details[input_map['url_features']]['index'], url_feats)

    # Warm-up run (Load instructions into CPU cache)
    interpreter.invoke()
    
    latencies = []
    
    # Start Benchmark Loop
    start_total = time.time()
    
    for _ in range(NUM_ITERATIONS):
        # We re-set tensors in the loop to simulate a real message stream
        interpreter.set_tensor(input_details[input_map['input_ids']]['index'], input_ids)
        interpreter.set_tensor(input_details[input_map['attention_mask']]['index'], attention_mask)
        interpreter.set_tensor(input_details[input_map['url_features']]['index'], url_feats)

        start_inf = time.perf_counter() # High precision timer
        interpreter.invoke()
        end_inf = time.perf_counter()
        
        latencies.append((end_inf - start_inf) * 1000) # Convert to ms
        
    end_total = time.time()
    
    # Calculate Metrics
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = NUM_ITERATIONS / (end_total - start_total)
    
    print(f"Total Time for {NUM_ITERATIONS} inferences: {end_total - start_total:.2f} seconds")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"p95 Tail Latency: {p95_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} msgs/sec")
    
    return avg_latency, p95_latency, throughput

# ==========================================
# 3. EXECUTE EXPERIMENTS
# ==========================================
if __name__ == "__main__":
    print("\nStarting OS Scheduling Benchmarks...")
    
    # Get all available logical cores on your machine
    all_cores = list(range(psutil.cpu_count(logical=True)))
    
    # EXPERIMENT A: Baseline (Default Windows Scheduler allowed to use all cores)
    avg_base, p95_base, thru_base = run_benchmark(
        mode_name="BASELINE (Default OS Scheduler)", 
        pinned_cores=all_cores
    )
    
    # EXPERIMENT B: Strict Affinity (Pinned to Core 0 only)
    avg_pin, p95_pin, thru_pin = run_benchmark(
        mode_name="STRICT AFFINITY (Pinned to Core 0)", 
        pinned_cores=[0]
    )
    
    # ==========================================
    # 4. FINAL REPORT
    # ==========================================
    print("\n" + "="*50)
    print("FINAL RESULTS FOR IEEE ARTICLE (SECTION IV)")
    print("="*50)
    print(f"{'Metric':<15} | {'Baseline':<12} | {'Pinned':<12} | {'Difference'}")
    print("-" * 50)
    print(f"{'Avg Latency':<15} | {avg_base:>8.2f} ms | {avg_pin:>8.2f} ms | {avg_pin - avg_base:>+6.2f} ms")
    print(f"{'p95 Latency':<15} | {p95_base:>8.2f} ms | {p95_pin:>8.2f} ms | {p95_pin - p95_base:>+6.2f} ms")
    print(f"{'Throughput':<15} | {thru_base:>8.2f} m/s | {thru_pin:>8.2f} m/s | {thru_pin - thru_base:>+6.2f} m/s")
    print("="*50)