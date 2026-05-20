import os
import random
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

MAX_LEN = 64  # Fixed sequence length
BATCH_SIZE = 32
EPOCHS = 3
MODEL_NAME = 'distilbert-base-uncased'
SEEDS = [42, 123, 2026]
SMS_DIR = 'sms_datasets/'
SHORTENER_DOMAINS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'is.gd', 'ow.ly'}
HOMOGLYPH_DICT = {
    'а': 'a', 'ϲ': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x', 'у': 'y',
    'с': 'c', 'в': 'v', 'к': 'k', 'н': 'n', 'т': 't', 'м': 'm',
    'α': 'a', 'ν': 'n', 'ρ': 'p',
}

sns.set_theme(style='whitegrid')

# ============================================================================
# SECURITY PREPROCESSING FUNCTIONS
# ============================================================================

def handle_zero_width(text):
    text = str(text)
    zero_width_pattern = re.compile(r'[\u200B-\u200D\uFEFF]')
    contains_zero_width = 1 if zero_width_pattern.search(text) else 0
    cleaned_text = zero_width_pattern.sub('', text)
    return cleaned_text, contains_zero_width


def refang_text(text):
    text = re.sub(r'\[\.\]|\(\.\)|\{\.\}', '.', text)
    text = re.sub(r'(?i)hxxp', 'http', text)
    return text


def normalize_homoglyphs(text):
    for homoglyph, standard in HOMOGLYPH_DICT.items():
        text = text.replace(homoglyph, standard)
    return text


def extract_urls_with_regex(text):
    url_pattern = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    )
    return url_pattern.findall(text)


def resolve_url_if_shortened(url):
    # Offline edge inference requires zero-latency processing.
    # We therefore enforce an offline-only pass-through for URL expansion.
    return url


def preprocess_smishing_message(raw_sms):
    text, has_zero_width = handle_zero_width(raw_sms)
    text = refang_text(text)
    text = normalize_homoglyphs(text)
    extracted_urls = extract_urls_with_regex(text)
    final_urls = [resolve_url_if_shortened(url) for url in extracted_urls]
    security_features = {
        'has_zero_width': has_zero_width,
        'url_count': len(final_urls),
        'has_shortened_url': 1 if any(any(domain in url for domain in SHORTENER_DOMAINS) for url in extracted_urls) else 0,
    }
    return text, final_urls, security_features


def normalize_text(text):
    text = str(text).lower()
    return normalize_homoglyphs(text)


def extract_url_features(text):
    cleaned_text, final_urls, security_features = preprocess_smishing_message(text)
    has_url = 1 if final_urls else 0
    num_urls = len(final_urls)
    has_zero_width = security_features['has_zero_width']
    has_shortened = security_features['has_shortened_url']
    return np.array([has_url, num_urls, has_zero_width, has_shortened], dtype=np.float32)


def mask_digits(text):
    text = str(text)
    return re.sub(r'\d', '0', text)


def load_and_harmonize():
    main_dfs = []
    transactional_dfs = []

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

    try:
        path = os.path.join(SMS_DIR, 'smishtank.csv')
        if os.path.exists(path):
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

    try:
        path = os.path.join(SMS_DIR, 'phishing.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'phishing' else 0)
            main_dfs.append(df[['text', 'label']])
            print(f"Loaded Kaggle Phishing: {len(df)} rows")
    except Exception as e:
        print(f"Skipped Kaggle Phishing: {e}")

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

    transactional_files = ['transactional_sms_dataset.csv', 'transactional_sms_dataset_500.csv']
    for file in transactional_files:
        try:
            path = os.path.join(SMS_DIR, file)
            if not os.path.exists(path):
                path = os.path.join(os.path.expanduser('~'), 'Downloads', file)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df.rename(columns={'message_body': 'text'})
                df['label'] = 0
                transactional_dfs.append(df[['text', 'label']])
                print(f"Loaded Synthetic Transactional {file}: {len(df)} rows")
        except Exception as e:
            print(f"Skipped {file}: {e}")

    bank_dir = os.path.join(os.getcwd(), 'bank_transactions')
    if not os.path.exists(bank_dir):
        bank_dir = SMS_DIR
    for file in ['debit.csv', 'credit.csv']:
        try:
            path = os.path.join(bank_dir, file)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df.rename(columns={'SMS': 'text'})
                df['label'] = 0
                transactional_dfs.append(df[['text', 'label']])
                print(f"Loaded Bank Transaction {file}: {len(df)} rows")
        except Exception as e:
            print(f"Skipped Bank {file}: {e}")

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

    if transactional_dfs:
        transactional_data = pd.concat(transactional_dfs, ignore_index=True)
        transactional_data = transactional_data.dropna(subset=['text', 'label'])
        transactional_data['text'] = transactional_data['text'].astype(str).str.strip()
        transactional_data = transactional_data[transactional_data['text'] != '']
        transactional_data = transactional_data.drop_duplicates(subset=['text'])

        print(f"\nTRANSACTIONAL HAM (before oversampling): {len(transactional_data)} samples")
        transactional_oversampled = pd.concat([transactional_data] * 10, ignore_index=True)
        print(f"TRANSACTIONAL HAM (after 10x oversampling): {len(transactional_oversampled)} samples")
        print("-" * 30)
        full_data = pd.concat([main_data, transactional_oversampled], ignore_index=True)
    else:
        print("No transactional ham files found. Using main dataset only.")
        full_data = main_data

    full_data = full_data.sample(frac=1.0, random_state=999).reset_index(drop=True)
    print(f"\nFINAL HARMONIZED DATASET: {len(full_data)} samples")
    print(full_data['label'].value_counts())
    print("-" * 30)
    return full_data


def encode_texts(texts, tokenizer):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='tf'
    )


def build_hybrid_model():
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    bert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)
    bert_model.trainable = False
    bert_output = bert_model(input_ids, attention_mask=input_mask)[0][:, 0, :]
    input_url = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name='url_features')
    concatenated = tf.keras.layers.Concatenate()([bert_output, input_url])
    dense = tf.keras.layers.Dense(64, activation='relu')(concatenated)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    model = tf.keras.Model(inputs=[input_ids, input_mask, input_url], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_unimodal_model():
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    input_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    bert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)
    bert_model.trainable = False
    bert_output = bert_model(input_ids, attention_mask=input_mask)[0][:, 0, :]
    dense = tf.keras.layers.Dense(64, activation='relu')(bert_output)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_class_weight(y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return {0: class_weights[0], 1: class_weights[1]}


def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Threat'], yticklabels=['Benign', 'Threat'])
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_training_curves(history, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)
    axes[0].plot(history.history['loss'], marker='o', label='Train Loss')
    axes[0].plot(history.history['val_loss'], marker='o', label='Validation Loss')
    axes[0].set_title('Hybrid Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history['accuracy'], marker='o', label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
    axes[1].set_title('Hybrid Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_adversarial_resilience_chart(unimodal_recall, hybrid_recall, filename):
    labels = ['Unimodal', 'Hybrid']
    recalls = [unimodal_recall * 100, hybrid_recall * 100]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.barplot(x=labels, y=recalls, palette=['gray', 'royalblue'], ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Recall (%)')
    ax.set_title('Adversarial Test Set Recall')
    for idx, value in enumerate(recalls):
        ax.text(idx, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }


def format_averages_table(results):
    summary = []
    models = ['Random Forest', 'Unimodal', 'Hybrid']
    for model in models:
        accuracies = [r[model]['accuracy'] for r in results]
        precisions = [r[model]['precision'] for r in results]
        recalls = [r[model]['recall'] for r in results]
        f1s = [r[model]['f1'] for r in results]
        summary.append({
            'Model': model,
            'Accuracy': np.mean(accuracies),
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'F1-Score': np.mean(f1s),
        })
    df = pd.DataFrame(summary)
    df[['Accuracy', 'Precision', 'Recall', 'F1-Score']] = df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].round(4)
    return df


def main():
    data = load_and_harmonize()
    data['text'] = data['text'].apply(mask_digits)
    data['text'] = data['text'].apply(normalize_text)
    data['url_features'] = data['text'].apply(extract_url_features)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    encodings = encode_texts(data['text'], tokenizer)
    X_ids = encodings['input_ids'].numpy()
    X_masks = encodings['attention_mask'].numpy()
    X_urls = np.stack(data['url_features'].values)
    labels = data['label'].values.astype(np.float32)

    run_results = []
    final_artifacts = None

    for run_seed in SEEDS:
        print(f"\n{'=' * 50}\nRUN SEED: {run_seed}\n{'=' * 50}")
        np.random.seed(run_seed)
        random.seed(run_seed)

        index_train, index_test, _, _ = train_test_split(
            data.index.to_numpy(),
            labels,
            test_size=0.2,
            random_state=run_seed,
            stratify=labels
        )

        X_train_ids = X_ids[index_train]
        X_test_ids = X_ids[index_test]
        X_train_masks = X_masks[index_train]
        X_test_masks = X_masks[index_test]
        X_train_urls = X_urls[index_train]
        X_test_urls = X_urls[index_test]
        y_train = labels[index_train]
        y_test = labels[index_test]

        X_train_text = data.loc[index_train, 'text']
        X_test_text = data.loc[index_test, 'text']

        class_weight_dict = get_class_weight(y_train)
        print(f"Using class weights: {class_weight_dict}")

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=run_seed)
        rf_model.fit(X_train_tfidf, y_train)
        rf_predictions = rf_model.predict(X_test_tfidf)
        rf_metrics = evaluate_metrics(y_test, rf_predictions)

        unimodal_model = build_unimodal_model()
        unimodal_history = unimodal_model.fit(
            {'input_ids': X_train_ids, 'attention_mask': X_train_masks},
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            verbose=1
        )
        uni_probs = unimodal_model.predict({'input_ids': X_test_ids, 'attention_mask': X_test_masks})
        uni_predictions = (uni_probs > 0.5).astype(int)
        unimodal_metrics = evaluate_metrics(y_test, uni_predictions)

        hybrid_model = build_hybrid_model()
        hybrid_history = hybrid_model.fit(
            {'input_ids': X_train_ids, 'attention_mask': X_train_masks, 'url_features': X_train_urls},
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            verbose=1
        )
        hybrid_probs = hybrid_model.predict({'input_ids': X_test_ids, 'attention_mask': X_test_masks, 'url_features': X_test_urls})
        hybrid_predictions = (hybrid_probs > 0.5).astype(int)
        hybrid_metrics = evaluate_metrics(y_test, hybrid_predictions)

        run_results.append({
            'Random Forest': rf_metrics,
            'Unimodal': unimodal_metrics,
            'Hybrid': hybrid_metrics,
        })

        if run_seed == SEEDS[-1]:
            final_artifacts = {
                'hybrid_history': hybrid_history,
                'rf_predictions': rf_predictions,
                'uni_predictions': uni_predictions,
                'hybrid_predictions': hybrid_predictions,
                'y_test': y_test,
                'X_test_text': X_test_text,
                'unimodal_model': unimodal_model,
                'hybrid_model': hybrid_model,
            }

    summary_df = format_averages_table(run_results)
    print("\n=== 3-Run Averages ===")
    print(summary_df.to_string(index=False))

    if final_artifacts is not None:
        print(f"\nGenerating final manuscript assets for seed {SEEDS[-1]}")
        plot_training_curves(final_artifacts['hybrid_history'], 'training_validation_curves.png')

        cm_rf = confusion_matrix(final_artifacts['y_test'], final_artifacts['rf_predictions'])
        cm_uni = confusion_matrix(final_artifacts['y_test'], final_artifacts['uni_predictions'])
        cm_hybrid = confusion_matrix(final_artifacts['y_test'], final_artifacts['hybrid_predictions'])
        plot_confusion_matrix(cm_rf, 'Random Forest Confusion Matrix', 'cm_random_forest.png')
        plot_confusion_matrix(cm_uni, 'Unimodal Confusion Matrix', 'cm_unimodal.png')
        plot_confusion_matrix(cm_hybrid, 'Hybrid Confusion Matrix', 'cm_hybrid.png')

        np.random.seed(42)
        random.seed(42)
        malicious_indices = np.where(final_artifacts['y_test'].flatten() == 1)[0]
        adv_test_indices = np.random.choice(malicious_indices, size=min(500, len(malicious_indices)), replace=False)
        adv_texts = final_artifacts['X_test_text'].iloc[adv_test_indices].copy().values
        adv_labels = final_artifacts['y_test'][adv_test_indices]

        HOMOGLYPH_ATTACK = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'y': 'у'}
        ZERO_WIDTH = '\u200B'

        def apply_obfuscation(text):
            text = str(text)
            for lat, cyr in HOMOGLYPH_ATTACK.items():
                if random.random() > 0.5:
                    text = text.replace(lat, cyr)
            words = text.split()
            if words:
                target_idx = random.randint(0, len(words) - 1)
                target_word = words[target_idx]
                if len(target_word) > 2:
                    mid = len(target_word) // 2
                    words[target_idx] = target_word[:mid] + ZERO_WIDTH + target_word[mid:]
                    text = ' '.join(words)
            return text

        corrupted_texts = np.array([apply_obfuscation(t) for t in adv_texts])
        uni_adv_encodings = tokenizer(corrupted_texts.tolist(), padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='tf')
        uni_adv_probs = final_artifacts['unimodal_model'].predict({'input_ids': uni_adv_encodings['input_ids'], 'attention_mask': uni_adv_encodings['attention_mask']})
        uni_adv_preds = (uni_adv_probs > 0.5).astype(int)

        hybrid_adv_cleaned_texts = []
        hybrid_adv_urls = []
        for text in corrupted_texts:
            cleaned, has_zw = handle_zero_width(text)
            cleaned = refang_text(cleaned)
            cleaned = normalize_homoglyphs(cleaned)
            hybrid_adv_cleaned_texts.append(cleaned)
            hybrid_adv_urls.append(extract_url_features(cleaned))

        hybrid_adv_encodings = tokenizer(hybrid_adv_cleaned_texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='tf')
        hybrid_adv_urls_tensor = np.stack(hybrid_adv_urls)
        hybrid_adv_probs = final_artifacts['hybrid_model'].predict({'input_ids': hybrid_adv_encodings['input_ids'], 'attention_mask': hybrid_adv_encodings['attention_mask'], 'url_features': hybrid_adv_urls_tensor})
        hybrid_adv_preds = (hybrid_adv_probs > 0.5).astype(int)

        adv_recall_uni = recall_score(adv_labels, uni_adv_preds)
        adv_recall_hybrid = recall_score(adv_labels, hybrid_adv_preds)
        plot_adversarial_resilience_chart(adv_recall_uni, adv_recall_hybrid, 'adversarial_resilience_chart.png')
        print('Saved adversarial_resilience_chart.png')

        adv_results_df = pd.DataFrame({
            'Adversarial_SMS_Text': corrupted_texts,
            'True_Label': adv_labels.flatten(),
            'Unimodal_Prediction': uni_adv_preds.flatten(),
            'Hybrid_Prediction': hybrid_adv_preds.flatten()
        })
        adv_results_df['Hybrid_Correct'] = adv_results_df['True_Label'] == adv_results_df['Hybrid_Prediction']
        adv_results_df['Hybrid_Won_Where_Uni_Failed'] = (
            (adv_results_df['True_Label'] == adv_results_df['Hybrid_Prediction']) &
            (adv_results_df['True_Label'] != adv_results_df['Unimodal_Prediction'])
        )
        adv_export_filename = 'SafeLink_Adversarial_Results.csv'
        adv_results_df.to_csv(adv_export_filename, index=False, encoding='utf-8')
        print(f"Success! Adversarial predictions exported to '{adv_export_filename}'.")

    return run_results


if __name__ == '__main__':
    main()