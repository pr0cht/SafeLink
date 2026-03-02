import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer

# --- 1. CONFIGURATION ---
MODEL_PATH = 'safelink_model.tflite'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64

# --- 2. SAMPLE MESSAGES ---
test_samples = [
    "USPS: Your package is on hold due to a missing street number. Please update your delivery info here: http://usps-redelivery-notice.com/track",
    "CHASE BANK: We detected an unauthorized login attempt from a new device. Secure your account immediately at https://bit.ly/chase-auth-99",
    "URGENT: Your Apple account was just charged $899.00 for a MacBook Pro. If you did not authorize this purchase, call Fraud Prevention immediately at 1-800-555-0198.",
    "FINAL NOTICE: Your vehicle warranty has expired. Reply 'RENEW' to speak with an agent or you will be subject to out-of-pocket repair costs.",
    "C0NGRATULATIONS! U came in 1st in our weekly Wal-mart draw! Cl!ck www.prize-winner-2026.xyz to cla1m ur $1000 giftcard.",
    "Hey man, are we still on for lunch tomorrow at 12? Let me know if you need me to pick you up.",
    "Your Amazon login verification code is 492011. Do not share this code with anyone.",
    "Doctor's Appointment Reminder: You are scheduled to see Dr. Smith on Mar 5 at 10:00 AM. Reply Y to confirm or N to cancel.",
    "Congratulations! You've been selected for a free cruise to the Bahamas! Call now to claim your prize: 1-800-555-1234.",
    "ALERT: Your PayPal account has been limited due to suspicious activity. Please verify your identity immediately at https://paypal-secure-login.com/verify to restore full access.",
    "Dear Customer, Your Verizon bill is due on March 15th. Please pay your bill to avoid service interruption. Visit https://verizon-bill-pay.com to make a payment or view your statement.",
    "check us out on g00gle.com.",
    "hey, can you check us out our team on CS on cs.nnoney?"
]

# --- 3. LOAD MODEL & TOKENIZER ---
print("Loading model and tokenizer...")
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

def extract_url_features(text):
    has_url = 1 if 'http' in text or 'www' in text else 0
    len_url = len([w for w in text.split() if 'http' in w or 'www' in w])
    is_shortened = 1 if 'bit.ly' in text or 'tinyurl' in text else 0
    return np.array([has_url, len_url, is_shortened], dtype=np.float32)

# --- 4. RUN INFERENCE ---
print("\n" + "="*80)
print(f"{'PREDICTION':<12} | {'CONFIDENCE':<10} | {'MESSAGE'}")
print("="*80)

for text in test_samples:
    # Tokenize
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='np')
    
    # Prepare Inputs
    input_ids = tokens['input_ids'].astype(np.int32)
    attention_mask = tokens['attention_mask'].astype(np.int32)
    url_feats = extract_url_features(text).reshape(1, 3)

    # Set Tensors
    interpreter.set_tensor(input_details[input_map['input_ids']]['index'], input_ids)
    interpreter.set_tensor(input_details[input_map['attention_mask']]['index'], attention_mask)
    interpreter.set_tensor(input_details[input_map['url_features']]['index'], url_feats)

    # Invoke
    interpreter.invoke()
    
    # Get Result
    prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
    pred_str = "❌ PHISHING" if prob > 0.5 else "✅ SAFE"
    
    clean_text = text.replace('\n', ' ')[:50] + "..."
    print(f"{pred_str:<12} | {prob:.4f}     | {clean_text}")