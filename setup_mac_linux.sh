#!/bin/bash
echo "================================================"
echo "  ReguAI — Local ML Setup (Mac / Linux)"
echo "  TF-IDF + Naive Bayes, No API Key Needed"
echo "================================================"
echo

# Step 1: Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 not found. Install from https://www.python.org"
    exit 1
fi
echo "[OK] Python3 found: $(python3 --version)"

# Step 2: Virtual environment
echo
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "[OK] Virtual environment activated."

# Step 3: Upgrade pip
echo
echo "[2/5] Upgrading pip..."
pip install --upgrade pip --quiet

# Step 4: Install packages
echo
echo "[3/5] Installing Flask, scikit-learn, NLTK, numpy..."
pip install flask flask-cors scikit-learn nltk numpy pandas --quiet
echo "[OK] Packages installed."

# Step 5: NLTK data
echo
echo "[4/5] Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
print('[OK] NLTK data downloaded.')
"

# Step 6: Database
echo
echo "[5/5] Setting up SQLite database..."
python3 database/db_setup.py

echo
echo "================================================"
echo "  Setup Complete!"
echo "  Run:  source venv/bin/activate && python3 app.py"
echo "  Open: http://localhost:5000"
echo "================================================"
