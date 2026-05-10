@echo off
echo ================================================
echo   ReguAI — Local ML Setup (Windows)
echo   TF-IDF + Naive Bayes, No API Key Needed
echo ================================================
echo.

:: Step 1: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed.
    echo Please install Python 3.9 or above from https://www.python.org
    pause
    exit /b 1
)
echo [OK] Python found.

:: Step 2: Create virtual environment
echo.
echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo [OK] Virtual environment created.

:: Step 3: Upgrade pip
echo.
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Step 4: Install packages
echo.
echo [3/5] Installing packages (Flask, scikit-learn, NLTK, spaCy)...
pip install flask flask-cors scikit-learn nltk numpy pandas --quiet
echo [OK] Packages installed.

:: Step 5: Download NLTK data
echo.
echo [4/5] Downloading NLTK language data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('punkt_tab', quiet=True); print('[OK] NLTK data downloaded.')"

:: Step 6: Setup database
echo.
echo [5/5] Creating SQLite database and seeding training data...
python database\db_setup.py

echo.
echo ================================================
echo   Setup Complete!
echo   Now run:  python app.py
echo   Then open: http://localhost:5000
echo ================================================
pause
