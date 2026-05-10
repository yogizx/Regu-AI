"""
app.py — ReguAI Flask Backend
Run with: python app.py
Serves at: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.ml_models import (
    NERAnonymiser,
    TFIDFSummariser,
    CompletenessChecker,
    NaiveBayesClassifier,
    save_document,
    save_results,
    get_history
)

app = Flask(__name__)
CORS(app)

# ── Initialise all models at startup ──
print("[ReguAI] Loading models...")
anonymiser  = NERAnonymiser()
summariser  = TFIDFSummariser()
checker     = CompletenessChecker()
classifier  = NaiveBayesClassifier()
classifier.train()
print("[ReguAI] All models ready. Starting server...")


# ════════════════════════════════════════
#  ROUTES — Pages
# ════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ════════════════════════════════════════
#  API ROUTE 1 — Anonymisation
# ════════════════════════════════════════

@app.route('/api/anonymise', methods=['POST'])
def api_anonymise():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc_id = save_document(text)
    anon_text, entities = anonymiser.anonymise(text)

    result = {"anonymised_text": anon_text, "entities": entities, "doc_id": doc_id}
    save_results('anonymisation_results', doc_id, result)

    return jsonify(result)


# ════════════════════════════════════════
#  API ROUTE 2 — Summarisation
# ════════════════════════════════════════

@app.route('/api/summarise', methods=['POST'])
def api_summarise():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc_id = save_document(text)
    result = summariser.summarise(text)
    result['doc_id'] = doc_id
    save_results('summarisation_results', doc_id, result)

    return jsonify(result)


# ════════════════════════════════════════
#  API ROUTE 3 — Completeness Check
# ════════════════════════════════════════

@app.route('/api/completeness', methods=['POST'])
def api_completeness():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc_id = save_document(text)
    result = checker.check(text)
    result['doc_id'] = doc_id
    save_results('completeness_results', doc_id, result)

    return jsonify(result)


# ════════════════════════════════════════
#  API ROUTE 4 — Severity Classification
# ════════════════════════════════════════

@app.route('/api/classify', methods=['POST'])
def api_classify():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc_id = save_document(text)
    result = classifier.classify(text)
    result['doc_id'] = doc_id
    save_results('classification_results', doc_id, result)

    return jsonify(result)


# ════════════════════════════════════════
#  API ROUTE 5 — Full Pipeline
# ════════════════════════════════════════

@app.route('/api/pipeline', methods=['POST'])
def api_pipeline():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    doc_id = save_document(text)

    # Step 1: Anonymise
    anon_text, entities = anonymiser.anonymise(text)
    anon_result = {"anonymised_text": anon_text, "entities": entities}
    save_results('anonymisation_results', doc_id, anon_result)

    # Step 2: Summarise (on anonymised text)
    sum_result = summariser.summarise(anon_text)
    save_results('summarisation_results', doc_id, sum_result)

    # Step 3: Completeness (on original text)
    comp_result = checker.check(text)
    save_results('completeness_results', doc_id, comp_result)

    # Step 4: Classify (on original text)
    class_result = classifier.classify(text)
    save_results('classification_results', doc_id, class_result)

    # Build reviewer action summary
    action_items = []
    label = class_result['label']
    if label in ('Death', 'Disability'):
        action_items.append(f"PRIORITY: Severity classified as {label}. Escalate immediately per CDSCO Schedule Y 7-day reporting timeline.")
    if comp_result['missing']:
        action_items.append(f"Return to reporter: Missing fields — {', '.join(comp_result['missing'][:5])}.")
    if comp_result['score'] >= 80:
        action_items.append(f"Report is {comp_result['score']}% complete. Proceed with medical review.")
    if sum_result['urgency'] == 'Immediate':
        action_items.append("Urgency: Immediate — Expedited regulatory timeline applies.")
    action_items.append("Archive anonymised copy per CDSCO data protection guidelines.")

    return jsonify({
        "doc_id": doc_id,
        "anonymisation": anon_result,
        "summarisation": sum_result,
        "completeness": comp_result,
        "classification": class_result,
        "action_summary": "\n\n".join(action_items)
    })


# ════════════════════════════════════════
#  API ROUTE 6 — History
# ════════════════════════════════════════

@app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify(get_history())


# ════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  ReguAI — CDSCO Document Intelligence")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
