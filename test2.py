import time
import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = 'safelink_model.tflite'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64
# We will test on UCI Spam first as it has both Safe and Phishing examples
DATA_PATH = 'sms_datasets/sms_spam.csv' 

# ==========================================
# 2. LOAD & PREPARE DATA
# ==========================================
print(f"Loading real data from {DATA_PATH}...")

try:
    # 1. Load the file
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    
    # DEBUG: Print the actual columns found so we can see what's wrong
    print(f"Found columns: {df.columns.tolist()}")

    # 2. Rename columns dynamically based on what we find
    # Scenario A: Standard UCI headers (v1, v2)
    if 'v2' in df.columns:
        df = df.rename(columns={'v2': 'text', 'v1': 'label'})
    
    # Scenario B: Your specific headers (sms, label)
    elif 'sms' in df.columns:
        df = df.rename(columns={'sms': 'text'})
    
    # Scenario C: Generic headers (message, text)
    elif 'message' in df.columns:
        df = df.rename(columns={'message': 'text'})
        
    # 3. Ensure 'text' column exists now
    if 'text' not in df.columns:
        raise ValueError(f"Could not find a text column! Available columns: {df.columns.tolist()}")

    # 4. Standardize Labels to 0/1
    # If the label column is strings like 'ham'/'spam', map them.
    # If it's already 0/1, this map might turn them to NaN, so we check first.
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'spam': 1, 'ham': 0, 'phishing': 1, 'safe': 0})
        # Fill any NaNs (in case of unexpected labels) with 0
        df['label'] = df['label'].fillna(0).astype(int)

    # 5. Sample the data
    test_df = df.sample(100, random_state=42) 
    print(f"Selected {len(test_df)} messages for testing.")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# ==========================================
# 3. SETUP TFLITE INTERPRETER
# ==========================================
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map inputs dynamically
input_map = {}
for i, detail in enumerate(input_details):
    if 'input_ids' in detail['name']: input_map['input_ids'] = i
    elif 'attention_mask' in detail['name']: input_map['attention_mask'] = i
    elif 'url_features' in detail['name']: input_map['url_features'] = i

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def extract_url_features(text):
    has_url = 1 if 'http' in text else 0
    len_url = len([w for w in text.split() if 'http' in w])
    is_shortened = 1 if 'bit.ly' in text or 'tinyurl' in text else 0
    return np.array([has_url, len_url, is_shortened], dtype=np.float32)

def predict(text):
    # Tokenize
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='np'
    )
    
    # Prepare Inputs
    input_ids = tokens['input_ids'].astype(np.int32)
    attention_mask = tokens['attention_mask'].astype(np.int32)
    url_feats = extract_url_features(text).reshape(1, 3)

    # Set Tensors
    interpreter.set_tensor(input_details[input_map['input_ids']]['index'], input_ids)
    interpreter.set_tensor(input_details[input_map['attention_mask']]['index'], attention_mask)
    interpreter.set_tensor(input_details[input_map['url_features']]['index'], url_feats)

    # Run Inference
    interpreter.invoke()
    
    # Get Result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = output_data[0][0]
    return prob, 1 if prob > 0.5 else 0

# ==========================================
# 5. RUN TEST LOOP (FULL OUTPUT)
# ==========================================
print("\nRunning Inference on 100 samples...")
results = []
errors = []

# Print Table Header
print(f"{'STATUS':<4} | {'PRED':<9} | {'ACTUAL':<9} | {'CONF':<6} | {'TEXT'}")
print("-" * 110)

for index, row in test_df.iterrows():
    text = str(row['text'])
    actual_label = row['label']
    
    # Run Model
    prob, pred_label = predict(text)
    
    # Formatting
    pred_str = "PHISHING" if pred_label == 1 else "SAFE"
    act_str = "PHISHING" if actual_label == 1 else "SAFE"
    
    # Check Result
    if pred_label == actual_label:
        status = "✅"
    else:
        status = "❌"
        errors.append({'text': text, 'actual': act_str, 'predicted': pred_str, 'probability': prob})

    # PRINT EVERYTHING (Truncate text to fit screen)
    clean_text = text.replace('\n', ' ')[:60] 
    print(f"{status:<4} | {pred_str:<9} | {act_str:<9} | {prob:.4f} | {clean_text}...")

    results.append(pred_label)

    results.append(pred_label)

# ==========================================
# 6. SAVE ERRORS TO CSV
# ==========================================
if errors:
    error_df = pd.DataFrame(errors)
    error_df.to_csv('error_analysis.csv', index=False)
    print(f"\n\n⚠️ Found {len(errors)} errors. Saved to 'error_analysis.csv'.")
    print("Check this file to see exactly which messages confused the AI.")
else:
    print("\n\n✅ AMAZING! No errors found in this sample batch.")

# ==========================================
# 7. FINAL METRICS
# ==========================================
y_true = test_df['label'].tolist()
print("\n" + "="*40)
print(classification_report(y_true, results, target_names=['Safe', 'Phishing']))
print("="*40)