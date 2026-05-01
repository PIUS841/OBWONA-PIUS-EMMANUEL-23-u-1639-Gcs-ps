"""
Train a Logistic Regression model on the phishing URL dataset
and save it as a .pkl file for use by the Flask backend.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "phishing_site_urls.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.pkl")
VEC_PATH = os.path.join(SCRIPT_DIR, "vectorizer.pkl")


def extract_url_features(url: str) -> str:
    """Return a string of URL tokens for TF-IDF."""
    tokens = re.split(r'[^a-zA-Z0-9]', url.lower())
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def download_dataset():
    """Download a sample phishing dataset if not exists."""
    import urllib.request
    
    print("📥 Downloading sample dataset...")
    # Alternative dataset URLs (try multiple sources)
    urls_to_try = [
        "https://raw.githubusercontent.com/siddharthdixit/Phishing-Url-Detection/master/Data/phishing_site_urls.csv",
        "https://raw.githubusercontent.com/faizann24/Phishing-Website-Detection-using-Machine-Learning/master/dataset.csv"
    ]
    
    for url in urls_to_try:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, CSV_PATH)
            print(f"✅ Dataset downloaded to {CSV_PATH}")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    print("\n❌ Could not download dataset automatically.")
    print("Please manually download a phishing dataset with 'URL' and 'Label' columns.")
    print("Example format:")
    print("  URL,Label")
    print("  https://google.com,good")
    print("  http://phishing-site.com,bad")
    return False


def create_sample_dataset():
    """Create a sample dataset for testing if no dataset is available."""
    print("📝 Creating sample dataset for demonstration...")
    
    sample_data = {
        'URL': [
            'https://www.google.com',
            'https://www.facebook.com',
            'https://www.amazon.com',
            'https://github.com',
            'https://stackoverflow.com',
            'http://paypal.com.login.secure.verify.com',
            'http://facebook.com.account-verify.secure.com/login',
            'http://192.168.1.1/bank/login',
            'https://secure-verify-paypal.com',
            'http://appleid-verify.xyz',
            'https://www.microsoft.com',
            'http://bankofamerica-verify.tk',
            'https://www.python.org',
            'http://instagram-login.ga',
            'https://www.reddit.com'
        ],
        'Label': [
            'good', 'good', 'good', 'good', 'good',
            'bad', 'bad', 'bad', 'bad', 'bad',
            'good', 'bad', 'good', 'bad', 'good'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Sample dataset created at {CSV_PATH}")
    return True


def main():
    print("\n" + "="*60)
    print("🤖 Scam URL Detector - Model Training")
    print("="*60 + "\n")
    
    # Load or download dataset
    print("[1/6] Loading dataset...")
    
    if not os.path.exists(CSV_PATH):
        print("Dataset not found.")
        choice = input("Do you want to (1) download dataset or (2) create sample dataset? (1/2): ")
        if choice == '1':
            if not download_dataset():
                create_sample_dataset()
        else:
            create_sample_dataset()
    
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Clean data
    print("\n[2/6] Cleaning data...")
    df = df.drop_duplicates()
    df = df.dropna(subset=["URL", "Label"])
    print(f"✅ After cleaning: {len(df):,} rows")
    
    # Check label values
    unique_labels = df["Label"].unique()
    print(f"📊 Labels found: {unique_labels}")
    
    # Binary encode labels
    label_mapping = {
        'bad': 1, 'phishing': 1, 'malicious': 1, 'scam': 1,
        'good': 0, 'legitimate': 0, 'benign': 0, 'safe': 0
    }
    
    df["label_bin"] = df["Label"].str.strip().str.lower().map(label_mapping)
    
    # Drop rows where mapping failed
    before_drop = len(df)
    df = df.dropna(subset=["label_bin"])
    print(f"✅ After label mapping: {len(df):,} rows (dropped {before_drop - len(df)} invalid rows)")
    
    if len(df) == 0:
        print("❌ No valid data found. Please check your dataset format.")
        sys.exit(1)
    
    print(f"\n📊 Class distribution:")
    print(f"   Legitimate (0): {(df['label_bin'] == 0).sum():,} ({(df['label_bin'] == 0).mean()*100:.1f}%)")
    print(f"   Scam (1):       {(df['label_bin'] == 1).sum():,} ({(df['label_bin'] == 1).mean()*100:.1f}%)")
    
    # Feature extraction
    print("\n[3/6] Extracting URL features...")
    df["url_tokens"] = df["URL"].apply(extract_url_features)
    
    X = df["url_tokens"]
    y = df["label_bin"]
    
    # Split data
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training set: {len(X_train):,} samples")
    print(f"✅ Test set: {len(X_test):,} samples")
    
    # TF-IDF Vectorization
    print("\n[5/6] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✅ Feature space dimension: {X_train_vec.shape[1]:,}")
    
    # Train model
    print("\n[6/6] Training Logistic Regression model...")
    model = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("📊 Model Evaluation")
    print("="*60)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Scam"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0,0]:,} (Legitimate correctly identified)")
    print(f"   False Positives: {cm[0,1]:,} (Legitimate flagged as scam)")
    print(f"   False Negatives: {cm[1,0]:,} (Scam missed)")
    print(f"   True Positives:  {cm[1,1]:,} (Scam correctly identified)")
    
    # Save model
    print("\n💾 Saving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Vectorizer saved to: {VEC_PATH}")
    
    # Test with sample URLs
    print("\n" + "="*60)
    print("🧪 Testing with sample URLs")
    print("="*60)
    
    test_urls = [
        "https://www.google.com",
        "https://github.com",
        "http://paypal.com.login.secure.verify.com",
        "http://facebook.com.account-verify.secure.com",
        "https://stackoverflow.com",
        "http://192.168.1.1/login/bank"
    ]
    
    for url in test_urls:
        tokens = extract_url_features(url)
        vec = vectorizer.transform([tokens])
        prob = model.predict_proba(vec)[0]
        label = "🚨 SCAM" if prob[1] > 0.5 else "✅ LEGITIMATE"
        print(f"{label:12} | {prob[1]*100:5.1f}% scam | {url[:60]}")
    
    print("\n" + "="*60)
    print("🎉 Training complete! You can now run: python app.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()