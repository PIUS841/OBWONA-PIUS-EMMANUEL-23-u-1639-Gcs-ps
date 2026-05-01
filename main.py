"""
Flask backend for the Scam URL Detector.
Serves the trained Logistic Regression model via a JSON API and
serves the static UI files.
"""

import os
import re
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from urllib.parse import urlparse

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create static directory if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)

# ── Load model and vectorizer ─────────────────────────────────────────────
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        log.info("✅ Model and vectorizer loaded successfully.")
        MODEL_READY = True
    else:
        log.warning("⚠️ Model files not found. Please run model.py first.")
        MODEL_READY = False
        model = vectorizer = None
except Exception as e:
    log.error(f"❌ Could not load model: {e}")
    MODEL_READY = False
    model = vectorizer = None


def extract_url_features(url: str) -> str:
    """Extract tokens from URL for TF-IDF."""
    # Add scheme if missing for better parsing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Split on non-alphanumeric characters
    tokens = re.split(r'[^a-zA-Z0-9]', url.lower())
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def url_stats(url: str) -> dict:
    """Return detailed statistics about a URL."""
    # Parse URL
    if not url.startswith(('http://', 'https://')):
        url_with_scheme = 'http://' + url
    else:
        url_with_scheme = url
    
    parsed = urlparse(url_with_scheme)
    domain = parsed.netloc if parsed.netloc else url.split('/')[0]
    
    # Path depth
    path_parts = [p for p in parsed.path.split('/') if p]
    depth = len(path_parts)
    
    # Check for IP address
    has_ip = bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain))
    
    # Count subdomains
    subdomain_count = max(domain.count('.') - 1, 0)
    
    # Check HTTPS
    has_https = url.lower().startswith('https')
    
    # URL length
    url_len = len(url)
    
    # Check for @ symbol
    has_at = '@' in url
    
    # Suspicious keywords
    suspicious_kws = ['login', 'signin', 'bank', 'update', 'secure', 'verify',
                      'account', 'password', 'paypal', 'ebay', 'amazon', 
                      'confirm', 'validate', 'authenticate', 'signin', 
                      'webscr', 'cgi-bin', 'redirect']
    kws_found = [kw for kw in suspicious_kws if kw in url.lower()]
    
    # Special characters count
    special_chars = len(re.findall(r'[^a-zA-Z0-9./:-]', url))
    
    # Dot count
    dot_count = url.count('.')
    
    # Check for abnormal TLDs
    abnormal_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club', '.live', '.online']
    has_abnormal_tld = any(tld in domain.lower() for tld in abnormal_tlds)
    
    # Check for multiple slashes
    has_multiple_slashes = '//' in url.replace('://', '')
    
    return {
        "url_length": url_len,
        "domain": domain,
        "subdomain_count": subdomain_count,
        "path_depth": depth,
        "has_https": has_https,
        "has_ip_address": has_ip,
        "has_at_symbol": has_at,
        "suspicious_kws": kws_found,
        "special_char_count": special_chars,
        "has_abnormal_tld": has_abnormal_tld,
        "dot_count": dot_count,
        "has_multiple_slashes": has_multiple_slashes
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if a URL is legitimate or a scam."""
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Please train first (python model.py)."}), 503
    
    data = request.get_json(force=True)
    urls = data.get("urls", [])
    
    if isinstance(urls, str):
        urls = [urls]
    
    if not urls:
        return jsonify({"error": "No URL(s) provided"}), 400
    
    results = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        
        try:
            # Extract features and predict
            tokens = extract_url_features(url)
            vec = vectorizer.transform([tokens])
            prob = model.predict_proba(vec)[0]  # [P(legit), P(scam)]
            label = model.predict(vec)[0]  # 0=legit, 1=scam
            confidence = float(np.max(prob)) * 100
            scam_prob = float(prob[1]) * 100
            
            results.append({
                "url": url,
                "verdict": "SCAM" if label == 1 else "LEGITIMATE",
                "scam_prob": round(scam_prob, 2),
                "legit_prob": round(100 - scam_prob, 2),
                "confidence": round(confidence, 2),
                "stats": url_stats(url),
            })
        except Exception as e:
            log.error(f"Error processing URL {url}: {e}")
            results.append({
                "url": url,
                "verdict": "ERROR",
                "scam_prob": 0,
                "legit_prob": 0,
                "confidence": 0,
                "stats": url_stats(url),
                "error": str(e)
            })
    
    return jsonify({"results": results})


@app.route('/api/status', methods=['GET'])
def status():
    """Check API and model status."""
    return jsonify({
        "model_ready": MODEL_READY,
        "model_type": type(model).__name__ if MODEL_READY else None,
        "api_version": "1.0.0",
        "status": "running"
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": MODEL_READY})


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    log.info("🚀 Starting Scam URL Detector on http://localhost:5000")
    log.info("📊 API endpoint: http://localhost:5000/api/predict")
    log.info("💚 Status check: http://localhost:5000/api/status")
    print("\n" + "="*60)
    print("🌟 Scam URL Detector is running!")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)