import os
import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MAX_LEN = 64  # Fixed sequence length [cite: 411]
BATCH_SIZE = 32
EPOCHS = 3
MODEL_NAME = 'distilbert-base-uncased'

def normalize_text(text):
    """
    De-obfuscation layer [cite: 366-367]
    Reverts common homoglyphs and cleans text.
    """
    # Simple example of homoglyph revert (expand this list)
    replacements = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c' # Cyrillic to Latin
    }
    text = str(text).lower()
    for cyr, lat in replacements.items():
        text = text.replace(cyr, lat)
    return text

def extract_url_features(text):
    """
    URL Branch Feature Extraction [cite: 402-404]
    Returns a numerical vector for the URL part.
    """
    # Placeholder: In reality, you'd regex extract the URL first
    has_url = 1 if 'http' in text else 0
    len_url = len([w for w in text.split() if 'http' in w])
    is_shortened = 1 if 'bit.ly' in text or 'tinyurl' in text else 0
    
    # Returning a simple 3-feature vector for demonstration
    return np.array([has_url, len_url, is_shortened], dtype=np.float32)

# ==========================================
# 3. DATA LOADING (HARMONIZATION LAYER)
# ==========================================

data = pd.DataFrame({
    'text': [
        "Urgent! Your account is locked. Click bit.ly/123", 
        "Hey, are we still meeting for lunch?", 
        "Win a free iPhone! Claim at http://scam.site"
    ],
    'label': [1, 0, 1]  # 1 = Malicious, 0 = Benign
})

# Apply Normalization
data['text'] = data['text'].apply(normalize_text)
data['url_features'] = data['text'].apply(extract_url_features)

# Prepare Inputs
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )

X_ids = encode_texts(data['text'])['input_ids']
X_masks = encode_texts(data['text'])['attention_mask']
X_urls = np.stack(data['url_features'].values)
y = data['label'].values

# Stratified Split [cite: 260]
X_train_ids, X_test_ids, X_train_masks, X_test_masks, X_train_urls, X_test_urls, y_train, y_test = train_test_split(
    X_ids.numpy(), X_masks.numpy(), X_urls, y, test_size=0.3, stratify=y
)

# ==========================================
# 4. HYBRID MODEL ARCHITECTURE [cite: 406-409]
# ==========================================
def build_hybrid_model():
    # --- Branch A: Text (DistilBERT) ---
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    
    bert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)
    bert_model.trainable = False  # Freeze BERT layers for speed
    
    # Get the [CLS] token embedding (768 dimensions) [cite: 400]
    bert_output = bert_model(input_ids, attention_mask=input_mask)[0][:, 0, :]
    
    # --- Branch B: URL Features ---
    input_url = tf.keras.layers.Input(shape=(3,), dtype=tf.float32, name='url_features')
    
    # --- Fusion Layer ---
    # Concatenate Semantic Vector (768) + URL Vector (3) [cite: 419]
    concatenated = tf.keras.layers.Concatenate()([bert_output, input_url])
    
    # Dense Layer + ReLU
    dense = tf.keras.layers.Dense(64, activation='relu')(concatenated)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    
    # Sigmoid Output [cite: 421]
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    
    model = tf.keras.Model(inputs=[input_ids, input_mask, input_url], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_hybrid_model()
model.summary()

# ==========================================
# 5. TRAINING
# ==========================================
print("Starting Training...")
history = model.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_masks, 'url_features': X_train_urls},
    y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ==========================================
# 6. TFLITE CONVERSION & QUANTIZATION [cite: 428-435]
# ==========================================
print("Converting to TFLite with Quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Dynamic Range Quantization (FP32 -> INT8)

tflite_model = converter.convert()

# Save the file
with open('safelink_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! 'safelink_model.tflite' generated.")