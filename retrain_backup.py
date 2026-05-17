import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import requests

MAX_LEN = 64  # Fixed sequence length 
BATCH_SIZE = 32
EPOCHS = 3
MODEL_NAME = 'distilbert-base-uncased'

# Known URL shortener domains
SHORTENER_DOMAINS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'is.gd', 'ow.ly'}

# Extended homoglyph dictionary (Cyrillic and Greek to Latin)
HOMOGLYPH_DICT = {
    # Cyrillic to Latin
    'а': 'a', 'ϲ': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x', 'у': 'y',
    'с': 'c', 'в': 'v', 'к': 'k', 'н': 'n', 'т': 't', 'м': 'm',
    # Greek to Latin
    'α': 'a', 'ν': 'n', 'ρ': 'p',
}

# ============================================================================
# SECURITY PREPROCESSING FUNCTIONS
# ============================================================================

def handle_zero_width(text):
    text = str(text)
    """
    Detect and remove zero-width characters (U+200B, U+200C, U+200D, U+FEFF).
    These invisible characters are used to break tokenization.
    
    Returns:
        tuple: (cleaned_text, contains_zero_width_flag)
    """
    zero_width_pattern = re.compile(r'[\u200B-\u200D\uFEFF]')
    contains_zero_width = 1 if zero_width_pattern.search(text) else 0
    cleaned_text = zero_width_pattern.sub('', text)
    return cleaned_text, contains_zero_width

def refang_text(text):
    """
    Refang defanged URLs for proper detection.
    Converts defanging patterns back to standard URL format:
    - [.], (.), {.} → .
    - hxxp/hxxps → http/https
    """
    # Replace defanged dots
    text = re.sub(r'\[\.\]|\(\.\)|\{\.\}', '.', text)
    # Replace defanged http
    text = re.sub(r'(?i)hxxp', 'http', text)
    return text

def normalize_homoglyphs(text):
    """
    Normalize homoglyphs (look-alike characters from different scripts)
    to their standard ASCII equivalents.
    """
    for homoglyph, standard in HOMOGLYPH_DICT.items():
        text = text.replace(homoglyph, standard)
    return text

def extract_urls_with_regex(text):
    """
    Extract URLs from text using regex pattern.
    """
    url_pattern = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    )
    return url_pattern.findall(text)

def resolve_url_if_shortened(url):
    """
    Resolve shortened URLs using HTTP HEAD requests.
    Returns the final destination URL or original if resolution fails.
    
    Args:
        url: The URL to check and potentially resolve
    
    Returns:
        The final unmasked URL (or original if resolution fails)
    """
    # Check if URL contains a known shortener domain
    if any(domain in url.lower() for domain in SHORTENER_DOMAINS):
        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=1.0,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            return response.url
        except requests.RequestException:
            # If HEAD request fails, return original URL
            return url
    return url

def preprocess_smishing_message(raw_sms):
    """
    Complete preprocessing pipeline for SMS text.
    Applies all security-focused transformations in sequence:
    1. Clean invisible characters
    2. Refang sneaky URLs
    3. Normalize fake characters
    4. Extract and resolve URLs
    
    Returns:
        tuple: (cleaned_text, final_urls, security_features_dict)
    """
    # 1. Handle zero-width characters
    text, has_zero_width = handle_zero_width(raw_sms)
    
    # 2. Refang URLs
    text = refang_text(text)
    
    # 3. Normalize homoglyphs
    text = normalize_homoglyphs(text)
    
    # 4. Extract URLs
    extracted_urls = extract_urls_with_regex(text)
    
    # 5. Resolve shortened URLs
    final_urls = [resolve_url_if_shortened(url) for url in extracted_urls]
    
    # Security features metadata
    security_features = {
        'has_zero_width': has_zero_width,
        'url_count': len(final_urls),
        'has_shortened_url': 1 if any(any(domain in url for domain in SHORTENER_DOMAINS) for url in extracted_urls) else 0,
    }
    
    return text, final_urls, security_features

# normalization
def normalize_text(text):
    """
    De-obfuscation layer using extended homoglyph dictionary.
    Reverts common homoglyphs and cleans text.
    """
    text = str(text).lower()
    text = normalize_homoglyphs(text)
    return text

# url feature extraction (enhanced with security features)
def extract_url_features(text):
    """
    Enhanced URL Branch Feature Extraction
    Returns a numerical vector for the URL part, including security indicators.
    """
    # Run the full preprocessing pipeline
    cleaned_text, final_urls, security_features = preprocess_smishing_message(text)
    
    # Extract features
    has_url = 1 if final_urls else 0
    num_urls = len(final_urls)
    has_zero_width = security_features['has_zero_width']
    has_shortened = security_features['has_shortened_url']
    
    # Returning a 4-feature vector: [has_url, num_urls, has_zero_width, has_shortened]
    return np.array([has_url, num_urls, has_zero_width, has_shortened], dtype=np.float32)

# data loading
SMS_DIR = 'sms_datasets/' 

def load_and_harmonize():
    main_dfs = []
    transactional_dfs = []
    
    # ========================================================================
    # MAIN DATASETS (UCI, Kaggle, SmishTank, etc.)
    # ========================================================================
    
    # --- 1. UCI SMS Spam (sms_spam.csv) ---
    try:
        path = os.path.join(SMS_DIR, 'sms_spam.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin-1')
            df = df.rename(columns={'sms': 'text', 'label': 'raw_label'})
            df['label'] = df['raw_label'].map({'ham': 0, 'spam': 1})
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded UCI Spam: {len(df)} rows")
    except Exception as e:
        print(f"Skipped UCI: {e}")

    # --- 2. WildGuard (wildguard.csv) ---
    try:
        path = os.path.join(SMS_DIR, 'wildguard.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={'prompt': 'text', 'label': 'raw_label'})
            df['label'] = df['raw_label'].apply(lambda x: 1 if str(x).lower() in ['harmful', 'malicious', 'true'] else 0)
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded WildGuard: {len(df)} rows")
    except Exception as e:
        print(f"Skipped WildGuard: {e}")

    # --- 3. SmishTank (smishtank.csv) WITH ENCODING FIX ---
    try:
        path = os.path.join(SMS_DIR, 'smishtank.csv')
        if os.path.exists(path):
            # FIX: Use latin-1 encoding to handle 0xb0 and other byte errors
            df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
            if 'MainText' in df.columns:
                df = df.rename(columns={'MainText': 'text'})
            elif 'Fulltext' in df.columns:
                df = df.rename(columns={'Fulltext': 'text'})
            df['label'] = 1
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded SmishTank: {len(df)} rows (All labeled as Malicious)")
    except Exception as e:
        print(f"Skipped SmishTank: {e}")

    # --- 4. Kaggle Phishing (phishing.csv) ---
    try:
        path = os.path.join(SMS_DIR, 'phishing.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'phishing' else 0)
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded Kaggle Phishing: {len(df)} rows")
    except Exception as e:
        print(f"Skipped Kaggle Phishing: {e}")

    # --- 5. Kaggle Smishing Eng (smishing_eng.csv) ---
    try:
        path = os.path.join(SMS_DIR, 'smishing_eng.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin-1')
            df = df.rename(columns={'v2': 'text', 'v1': 'raw_label'})
            df['label'] = df['raw_label'].map({'spam': 1, 'ham': 0})
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded Smishing Eng: {len(df)} rows")
    except Exception as e:
        print(f"Skipped Smishing Eng: {e}")

    # --- 6. Combined Dataset (combined_label_dataset.csv) ---
    try:
        path = os.path.join(SMS_DIR, 'combined_label_dataset.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={'message': 'text'})
            df['label'] = pd.to_numeric(df['smishing label'], errors='coerce').fillna(0).astype(int)
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded Combined Dataset: {len(df)} rows")
    except Exception as e:
        print(f"Skipped Combined Dataset: {e}")

    # ========================================================================
    # TRANSACTIONAL HAM DATASETS (Synthetic + Bank CSVs)
    # ========================================================================
    
    # Synthetic Transactional CSVs
    transactional_files = ['transactional_sms_dataset.csv', 'transactional_sms_dataset_500.csv']
    for file in transactional_files:
        try:
            path = os.path.join(SMS_DIR, file)
            if not os.path.exists(path):
                path = os.path.join(os.path.expanduser('~'), 'Downloads', file)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df.rename(columns={'message_body': 'text'})
                df['label'] = 0  # Benign
                transactional_dfs.append(df[['text', 'label']])
                print(f"Loaded Synthetic Transactional {file}: {len(df)} rows")
        except Exception as e:
            print(f"Skipped {file}: {e}")

    # Real Bank Transaction CSVs
    bank_dir = os.path.join(os.getcwd(), 'bank_transactions')
    if not os.path.exists(bank_dir):
        bank_dir = SMS_DIR
    bank_files = ['debit.csv', 'credit.csv']
    for file in bank_files:
        try:
            path = os.path.join(bank_dir, file)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df.rename(columns={'SMS': 'text'})
                df['label'] = 0  # Benign
                transactional_dfs.append(df[['text', 'label']])
                print(f"Loaded Bank Transaction {file}: {len(df)} rows")
        except Exception as e:
            print(f"Skipped Bank {file}: {e}")

    # ========================================================================
    # MERGE MAIN DATASET
    # ========================================================================
    if not main_dfs:
        raise ValueError("No main datasets loaded! Check folder path.")
    
    main_data = pd.concat(main_dfs, ignore_index=True)
    main_data = main_data.dropna(subset=['text', 'label'])
    main_data['text'] = main_data['text'].astype(str).str.strip()
    main_data = main_data[main_data['text'] != '']
    main_data = main_data.drop_duplicates(subset=['text'])
    
    print("-" * 30)
    print(f"MAIN DATASET: {len(main_data)} samples")
    print(main_data['label'].value_counts())
    print("-" * 30)

    # ========================================================================
    # MERGE TRANSACTIONAL HAM AND OVERSAMPLE BY 10x
    # ========================================================================
    if transactional_dfs:
        transactional_data = pd.concat(transactional_dfs, ignore_index=True)
        transactional_data = transactional_data.dropna(subset=['text', 'label'])
        transactional_data['text'] = transactional_data['text'].astype(str).str.strip()
        transactional_data = transactional_data[transactional_data['text'] != '']
        transactional_data = transactional_data.drop_duplicates(subset=['text'])
        
        print(f"\nTRANSACTIONAL HAM (before oversampling): {len(transactional_data)} samples")
        
        # OVERSAMPLE TRANSACTIONAL HAM BY 10x
        transactional_oversampled = pd.concat([transactional_data] * 10, ignore_index=True)
        print(f"TRANSACTIONAL HAM (after 10x oversampling): {len(transactional_oversampled)} samples")
        print("-" * 30)
        
        # COMBINE MAIN + OVERSAMPLED TRANSACTIONAL
        full_data = pd.concat([main_data, transactional_oversampled], ignore_index=True)
    else:
        print("No transactional ham files found. Using main dataset only.")
        full_data = main_data

    # Shuffle the combined dataset
    full_data = full_data.sample(frac=1.0, random_state=999).reset_index(drop=True)
    
    print(f"\nFINAL HARMONIZED DATASET: {len(full_data)} samples")
    print(full_data['label'].value_counts())
    print("-" * 30)
    
    return full_data


# Execute Loading
data = load_and_harmonize()

# ============================================================================
# DIGIT MASKING: Replace all digits (\d) with 0
# ============================================================================
print("\n" + "="*50)
print("STEP: Masking all digits in text (OTP codes, account numbers)...")
print("="*50)

def mask_digits(text):
    """
    Replace all digits (\d) with 0 to prevent the model from memorizing
    random OTP codes and account numbers.
    Example: "Your code is 650508" -> "Your code is 000000"
    """
    text = str(text)
    return re.sub(r'\d', '0', text)

data['text'] = data['text'].apply(mask_digits)
print(f"Digit masking complete on {len(data)} samples")
print("-" * 50)

# Apply Enhanced Preprocessing Pipeline
print("\nApplying security preprocessing...")
data['text'] = data['text'].apply(normalize_text)
data['url_features'] = data['text'].apply(extract_url_features)
print("Security preprocessing complete!")

# tokenization test samples
print("tokenizing data")

# Initialize Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )

# 1. Convert Text to BERT Tokens
encodings = encode_texts(data['text'])
X_ids = encodings['input_ids']
X_masks = encodings['attention_mask']

# 2. Convert URL Features to Numpy Array
# Note: Now includes [has_url, num_urls, has_zero_width, has_shortened] (4 features)
X_urls = np.stack(data['url_features'].values)

# 3. Get Labels
y = data['label'].values.astype(np.float32)

# 4. Perform Stratified Train/Test Split
print("Splitting data...")
X_train_ids, X_test_ids, X_train_masks, X_test_masks, X_train_urls, X_test_urls, y_train, y_test = train_test_split(
    X_ids.numpy(), 
    X_masks.numpy(), 
    X_urls, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training Samples: {len(X_train_ids)}")
print(f"Testing Samples: {len(X_test_ids)}")

# hybrid model architecture
def build_hybrid_model():
    # --- Branch A: Text (DistilBERT) ---
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    bert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)
    bert_model.trainable = False  # Freeze BERT layers for speed
    
    # Get the token embedding (768 dimensions)
    bert_output = bert_model(input_ids, attention_mask=input_mask)[0][:, 0, :]
    
    # --- Branch B: URL Features (now 4-dimensional: has_url, num_urls, has_zero_width, has_shortened) ---
    input_url = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name='url_features')
    
    # --- Fusion Layer ---
    # Concatenate Semantic Vector (768) + URL Vector (4)
    concatenated = tf.keras.layers.Concatenate()([bert_output, input_url])
    
    # Dense Layer + ReLU
    dense = tf.keras.layers.Dense(64, activation='relu')(concatenated)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    
    # Sigmoid Output 
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    
    model = tf.keras.Model(inputs=[input_ids, input_mask, input_url], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_hybrid_model()
model.summary()

# training
print("Starting Training...")

# Calculate class weights for balancing
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Using class weights: {class_weight_dict}")

history = model.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_masks, 'url_features': X_train_urls},
    y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict
)

# tflite conversion / quantization
print("Converting to TFLite with Quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Dynamic Range Quantization (FP32 -> INT8)

tflite_model = converter.convert()

# save the file
if os.path.exists('safelink_model.tflite'):
    os.remove('safelink_model.tflite')
with open('safelink_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! 'safelink_model.tflite' generated.")

# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("\n" + "="*50)
print("STARTING OBJECTIVE 1 TESTING PHASE")
print("="*50)

# --- Ensure we have the raw text split for the Random Forest Baseline ---
# We use the exact same random_state=42 so the test samples are identical to the neural network
X_train_text, X_test_text, _, _ = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# ==========================================
# BASELINE 1: Random Forest (TF-IDF)
# ==========================================
print("\nTraining Baseline 1: Random Forest (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

rf_predictions = rf_model.predict(X_test_tfidf)

# ==========================================
# BASELINE 2: Unimodal DistilBERT (Text Only)
# ==========================================
print("\nTraining Baseline 2: Unimodal DistilBERT (Text Only)...")
def build_unimodal_model():
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    bert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)
    bert_model.trainable = False  
    
    bert_output = bert_model(input_ids, attention_mask=input_mask)[0][:, 0, :]
    
    dense = tf.keras.layers.Dense(64, activation='relu')(bert_output)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    
    uni_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)
    uni_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return uni_model

unimodal_model = build_unimodal_model()
# Train quickly (you can lower epochs here just for the baseline if needed)
unimodal_model.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_masks},
    y_train, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1
)

# Predict probabilities and convert to binary classes (0 or 1)
uni_probs = unimodal_model.predict({'input_ids': X_test_ids, 'attention_mask': X_test_masks})
uni_predictions = (uni_probs > 0.5).astype(int)

# ==========================================
# THE PROPOSED SYSTEM: SafeLink Hybrid
# ==========================================
print("\nEvaluating Proposed System: SafeLink Hybrid...")
# We use the 'model' you already trained earlier in your script
hybrid_probs = model.predict({'input_ids': X_test_ids, 'attention_mask': X_test_masks, 'url_features': X_test_urls})
hybrid_predictions = (hybrid_probs > 0.5).astype(int)

# ==========================================
# METRICS & CONFUSION MATRICES
# ==========================================
def print_metrics_and_plot_cm(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Threat'], yticklabels=['Benign', 'Threat'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'cm_{model_name.replace(" ", "_")}.png')
    plt.close()
    print(f"Saved Confusion Matrix image: cm_{model_name.replace(' ', '_')}.png")

print_metrics_and_plot_cm(y_test, rf_predictions, "Baseline 1 - Random Forest")
print_metrics_and_plot_cm(y_test, uni_predictions, "Baseline 2 - Unimodal DistilBERT")
print_metrics_and_plot_cm(y_test, hybrid_predictions, "Proposed - SafeLink Hybrid")

print("\nTesting Phase Complete! Check your folder for the Confusion Matrix PNG files.")

print("\nGenerating Error Analysis File...")

# 1. Create a Pandas DataFrame to hold all the test data side-by-side
results_df = pd.DataFrame({
    'SMS_Text': X_test_text.values,
    'True_Label': y_test.flatten(),  # 0 = Benign, 1 = Threat
    'RF_Prediction': rf_predictions.flatten(),
    'Unimodal_Prediction': uni_predictions.flatten(),
    'Hybrid_Prediction': hybrid_predictions.flatten()
})

# 2. Add Helper Columns to easily filter your Excel file later!
# Was the Hybrid model correct?
results_df['Hybrid_Correct'] = results_df['True_Label'] == results_df['Hybrid_Prediction']

# Did the Hybrid model catch a threat that the Text-Only model MISSED?
# (This is exactly what you need to prove Objective 1 in your paper)
results_df['Hybrid_Won_Where_Uni_Failed'] = (
    (results_df['True_Label'] == results_df['Hybrid_Prediction']) & 
    (results_df['True_Label'] != results_df['Unimodal_Prediction'])
)

# 3. Export to CSV (You can open this directly in Microsoft Excel)
export_filename = "SafeLink_Test_Results.csv"
results_df.to_csv(export_filename, index=False, encoding='utf-8')

print(f"Success! All predictions exported to '{export_filename}'.")
print("You can open this file in Excel to review every single message.")


# ==========================================
# ADVERSARIAL RESILIENCE TESTING
# ==========================================

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

print("\n" + "="*50)
print("STARTING OBJECTIVE 2: ADVERSARIAL RESILIENCE TESTING")
print("="*50)

# --- 1. Generate the Adversarial Test Set ---
# We take only the malicious samples from our existing X_test set to see if we can trick the models into missing them.
malicious_indices = np.where(y_test.flatten() == 1)[0]

# Pick a random subset (e.g., 500 samples) to act as our adversarial attack group
np.random.seed(42) # For reproducibility
adv_test_indices = np.random.choice(malicious_indices, size=min(500, len(malicious_indices)), replace=False)

adv_texts = X_test_text.iloc[adv_test_indices].copy().values
adv_labels = y_test[adv_test_indices] # These are all 1 (Threat)

# --- 2. The Attacker: Apply Obfuscation ---
HOMOGLYPH_ATTACK = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'y': 'у'} # Latin to Cyrillic
ZERO_WIDTH = '\u200B'

def apply_obfuscation(text):
    text = str(text)
    # Attack 1: Homoglyph Injection (swap random letters)
    for lat, cyr in HOMOGLYPH_ATTACK.items():
        if random.random() > 0.5: # 50% chance to swap each specific character type
            text = text.replace(lat, cyr)
            
    # Attack 2: Zero-Width Injection (break up a word)
    words = text.split()
    if words:
        target_idx = random.randint(0, len(words)-1)
        target_word = words[target_idx]
        if len(target_word) > 2:
            # Insert invisible space in the middle of a random word
            mid = len(target_word) // 2
            words[target_idx] = target_word[:mid] + ZERO_WIDTH + target_word[mid:]
            text = " ".join(words)
    return text

# Apply the attack to create the corrupted dataset!
print("Applying adversarial obfuscation (Homoglyphs & Zero-Width) to test samples...")
corrupted_texts = np.array([apply_obfuscation(t) for t in adv_texts])

# --- 3. Evaluate Unimodal DistilBERT (No pre-processing pipeline) ---
print("\nEvaluating Baseline 2: Unimodal DistilBERT on Adversarial Data...")
# We must tokenize the corrupted text EXACTLY as the raw model would receive it
uni_adv_encodings = tokenizer(corrupted_texts.tolist(), padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='tf')
uni_adv_probs = unimodal_model.predict({'input_ids': uni_adv_encodings['input_ids'], 'attention_mask': uni_adv_encodings['attention_mask']})
uni_adv_preds = (uni_adv_probs > 0.5).astype(int)

# --- 4. Evaluate SafeLink Hybrid (WITH Pre-processing pipeline) ---
print("Evaluating Proposed System: SafeLink Hybrid on Adversarial Data...")
# SafeLink uses its pipeline to clean the text and extract features before prediction
hybrid_adv_cleaned_texts = []
hybrid_adv_urls = []

for text in corrupted_texts:
    # 1. Pipeline cleans the text (removes zero-width, reverts homoglyphs)
    cleaned, has_zw = handle_zero_width(text)
    cleaned = refang_text(cleaned)
    cleaned = normalize_homoglyphs(cleaned)
    hybrid_adv_cleaned_texts.append(cleaned)
    
    # 2. Extract features (It should catch the Zero-Width flag!)
    # We must format it exactly as a Pandas row to use your existing extract_url_features function
    row = pd.Series({'text': cleaned, 'has_zero_width': has_zw})
    url_feat = extract_url_features(row)
    hybrid_adv_urls.append(url_feat)

# Tokenize the CLEANED text
hybrid_adv_encodings = tokenizer(hybrid_adv_cleaned_texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='tf')
hybrid_adv_urls_tensor = np.stack(hybrid_adv_urls)

# Predict
hybrid_adv_probs = model.predict({'input_ids': hybrid_adv_encodings['input_ids'], 'attention_mask': hybrid_adv_encodings['attention_mask'], 'url_features': hybrid_adv_urls_tensor})
hybrid_adv_preds = (hybrid_adv_probs > 0.5).astype(int)

# --- 5. Output Results ---
print("\n--- ADVERSARIAL RESILIENCE RESULTS ---")
print("Baseline 2 - Unimodal (Text Only, No Pipeline):")
print(f"  Recall (Threats Caught): {recall_score(adv_labels, uni_adv_preds)*100:.2f}%")

print("\nProposed - SafeLink Hybrid (With Defense Pipeline):")
print(f"  Recall (Threats Caught): {recall_score(adv_labels, hybrid_adv_preds)*100:.2f}%")
print("---------------------------------------")
print("Note: Because all samples in this test are Threat (1), Accuracy equals Recall.")

print("\nGenerating Adversarial Error Analysis File...")

# 1. Create a Pandas DataFrame for the Adversarial Test Set
adv_results_df = pd.DataFrame({
    'Adversarial_SMS_Text': corrupted_texts,
    'True_Label': adv_labels.flatten(),  # These are all 1 (Threat)
    'Unimodal_Prediction': uni_adv_preds.flatten(),
    'Hybrid_Prediction': hybrid_adv_preds.flatten()
})

# 2. Add Helper Columns to easily filter your Excel file later
# Was the Hybrid model correct?
adv_results_df['Hybrid_Correct'] = adv_results_df['True_Label'] == adv_results_df['Hybrid_Prediction']

# Did the Hybrid model catch the adversarial threat that the Unimodal model MISSED?
adv_results_df['Hybrid_Won_Where_Uni_Failed'] = (
    (adv_results_df['True_Label'] == adv_results_df['Hybrid_Prediction']) & 
    (adv_results_df['True_Label'] != adv_results_df['Unimodal_Prediction'])
)

# 3. Export to CSV
adv_export_filename = "SafeLink_Adversarial_Results.csv"
adv_results_df.to_csv(adv_export_filename, index=False, encoding='utf-8')

print(f"Success! Adversarial predictions exported to '{adv_export_filename}'.")