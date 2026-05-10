"""
models/ml_models.py
Core ML models for ReguAI:
  - Model 1: NER Anonymiser (regex + spaCy patterns)
  - Model 2: TF-IDF Summariser (extractive)
  - Model 3: Completeness Checker (rule-based)
  - Model 4: TF-IDF + Naive Bayes Classifier
"""

import re
import os
import json
import sqlite3
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "reguai.db")

# Download NLTK data (only first time)
def download_nltk():
    for pkg in ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

download_nltk()


# ═══════════════════════════════════════════════════
#  MODEL 1 — NER ANONYMISER
#  Uses regex patterns + keyword matching to detect
#  and mask PII/PHI in medical documents
# ═══════════════════════════════════════════════════

class NERAnonymiser:
    def __init__(self):
        # Regex patterns for structured PII
        self.patterns = [
            # Phone numbers (Indian formats)
            (r'\+91[-\s]?\d{5}[-\s]?\d{5}', '[PHONE]'),
            (r'\b[6-9]\d{9}\b', '[PHONE]'),
            (r'\b\d{3,5}[-\s]\d{6,8}\b', '[PHONE]'),
            # Email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            # Dates of birth / dates (DD-Mon-YYYY, DD/MM/YYYY)
            (r'\b\d{1,2}[-/](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/]\d{2,4}\b', '[DATE]'),
            (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '[DATE]'),
            # Indian PIN codes
            (r'\b[1-9][0-9]{5}\b', '[PINCODE]'),
            # Patient/Report IDs
            (r'\bCDSCO/[A-Z0-9/\-]+\b', '[REPORT_ID]'),
            (r'\bCTRI/[0-9/]+\b', '[TRIAL_ID]'),
            (r'\bPT-\d+\b', '[PATIENT_ID]'),
            (r'\bCT-\d+\b', '[SUBJECT_ID]'),
            # Registration numbers
            (r'\bMCI-\d{4}-\d+\b', '[REG_NUMBER]'),
            (r'\bReg\s*No[:\s]+[A-Z0-9-]+\b', '[REG_NUMBER]'),
        ]

        # Indian hospital name keywords
        self.hospital_keywords = [
            'AIIMS', 'Apollo', 'Fortis', 'Max Hospital', 'Wockhardt',
            'Manipal', 'Medanta', 'Narayana', 'Lilavati', 'Breach Candy',
            'KEM Hospital', 'Safdarjung', 'PGI', 'CMC', 'JIPMER',
            'City Hospital', 'District Hospital', 'General Hospital',
            'Medical College', 'Institute of Medical'
        ]

        # Indian city/state names to catch addresses
        self.location_keywords = [
            'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad',
            'Pune', 'Ahmedabad', 'Bhopal', 'Nashik', 'Jaipur', 'Lucknow',
            'Nagar', 'Vihar', 'Maharashtra', 'Gujarat', 'Rajasthan', 'Pradesh',
            'Karnataka', 'Tamil Nadu', 'Telangana', 'West Bengal'
        ]

    def _mask_regex(self, text):
        """Apply all regex-based masking."""
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _detect_names(self, text):
        """
        Detect person names using:
        1. Titles (Dr., Mr., Mrs., Ms.) followed by capitalized words
        2. 'Patient Name:' / 'Name:' label patterns
        3. 'Patient:' prefix patterns
        """
        entities = []

        # Title-based name detection
        title_pattern = r'\b(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})'
        for m in re.finditer(title_pattern, text):
            entities.append(('PERSON_NAME', m.group(0)))

        # Label-based: "Patient Name: Rajesh Kumar"
        label_pattern = r'(?:Patient\s+Name|Name|Patient)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
        for m in re.finditer(label_pattern, text):
            entities.append(('PATIENT_NAME', m.group(1)))

        # "Reporter: Dr. Name" or "Physician: Name"
        reporter_pattern = r'(?:Reporter|Physician|Investigator|Consultant)\s*:\s*(?:Dr\.\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
        for m in re.finditer(reporter_pattern, text):
            entities.append(('DOCTOR_NAME', m.group(1)))

        return entities

    def _mask_names(self, text, entities):
        """Replace detected names in text."""
        for etype, name in entities:
            replacement = f'[{etype}]'
            text = text.replace(name, replacement)
        return text

    def _detect_hospitals(self, text):
        """Detect hospital names using keyword matching."""
        found = []
        for kw in self.hospital_keywords:
            # Find the full hospital phrase containing this keyword
            pattern = rf'[A-Z][a-zA-Z\s]*{re.escape(kw)}[a-zA-Z\s,]*(?:Hospital|Institute|Centre|Center|Clinic)?'
            for m in re.finditer(pattern, text):
                name = m.group(0).strip().rstrip(',')
                if len(name) > 3:
                    found.append(name)
        return found

    def _mask_hospitals(self, text, hospitals):
        for h in hospitals:
            text = text.replace(h, '[HOSPITAL_NAME]')
        return text

    def anonymise(self, text):
        """Full anonymisation pipeline. Returns anonymised text + entity list."""
        entities_found = []
        original = text

        # Step 1: Detect names before regex (so we capture full names first)
        name_entities = self._detect_names(text)
        hospital_names = self._detect_hospitals(text)

        # Step 2: Mask regex patterns
        text = self._mask_regex(text)

        # Step 3: Record and mask names
        for etype, name in name_entities:
            if name in original:
                entities_found.append({"type": etype, "original": name, "replaced_with": f"[{etype}]"})
        text = self._mask_names(text, name_entities)

        # Step 4: Mask hospitals
        for h in hospital_names:
            if h in text:
                entities_found.append({"type": "HOSPITAL_NAME", "original": h, "replaced_with": "[HOSPITAL_NAME]"})
        text = self._mask_hospitals(text, hospital_names)

        # Step 5: Detect all replaced placeholders from regex for entity list
        placeholder_map = {
            '[PHONE]': 'PHONE', '[EMAIL]': 'EMAIL', '[DATE]': 'DATE',
            '[PINCODE]': 'PINCODE', '[REPORT_ID]': 'REPORT_ID',
            '[TRIAL_ID]': 'TRIAL_ID', '[PATIENT_ID]': 'PATIENT_ID',
            '[REG_NUMBER]': 'REG_NUMBER'
        }
        for ph, etype in placeholder_map.items():
            count = text.count(ph)
            if count > 0:
                entities_found.append({"type": etype, "original": f"({count} instance(s) masked)", "replaced_with": ph})

        return text, entities_found


# ═══════════════════════════════════════════════════
#  MODEL 2 — TF-IDF EXTRACTIVE SUMMARISER
#  Scores sentences by TF-IDF weight and selects top N
# ═══════════════════════════════════════════════════

class TFIDFSummariser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Medical domain-specific important terms (boosted weight)
        self.medical_boost = {
            'adverse', 'event', 'serious', 'death', 'disability',
            'hospitali', 'causality', 'drug', 'patient', 'dose',
            'reaction', 'outcome', 'safety', 'efficacy', 'trial',
            'approval', 'review', 'missing', 'compliance', 'report'
        }

    def _clean(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:()\-/]', '', text)
        return text.strip()

    def _score_sentence(self, sentence, tfidf_scores, word_list):
        """Score a sentence based on TF-IDF weights of its words."""
        words = word_tokenize(sentence.lower())
        score = 0
        count = 0
        for w in words:
            if w in tfidf_scores:
                s = tfidf_scores[w]
                # Boost medical terms
                if any(w.startswith(b) for b in self.medical_boost):
                    s *= 1.5
                score += s
                count += 1
        return score / max(count, 1)

    def _detect_urgency(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ['death', 'fatal', 'died', 'deceased', 'mortality']):
            return 'Immediate'
        if any(w in text_lower for w in ['hospitali', 'disability', 'serious', 'icu', 'emergency']):
            return 'Routine'
        return 'Informational'

    def _detect_doc_type(self, text):
        text_lower = text.lower()
        if 'serious adverse event' in text_lower or 'sae' in text_lower:
            return 'SAE Report'
        if 'clinical trial' in text_lower or 'ctri' in text_lower:
            return 'Clinical Trial Report'
        if 'new drug' in text_lower or 'approval' in text_lower or 'nda' in text_lower:
            return 'Drug Approval Application'
        return 'Regulatory Document'

    def summarise(self, text, num_sentences=4):
        """Extract top N sentences using TF-IDF scoring."""
        text = self._clean(text)
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            summary = ' '.join(sentences)
            key_points = sentences[:4]
            return {
                "summary": summary,
                "key_points": key_points,
                "urgency": self._detect_urgency(text),
                "document_type": self._detect_doc_type(text),
                "sentence_count": len(sentences)
            }

        # Build TF-IDF matrix over all sentences
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=200
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            # Aggregate TF-IDF scores per word
            tfidf_scores = {}
            for i, fname in enumerate(feature_names):
                tfidf_scores[fname] = float(np.max(tfidf_matrix[:, i].toarray()))
        except:
            tfidf_scores = {}

        # Score each sentence
        scored = []
        words = list(tfidf_scores.keys())
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, tfidf_scores, words)
            # Prefer sentences from beginning and end (regulatory docs structure)
            position_bonus = 0.2 if i < 3 else (0.1 if i >= len(sentences) - 3 else 0)
            scored.append((score + position_bonus, i, sent))

        # Sort by score, keep order
        top = sorted(scored, reverse=True)[:num_sentences]
        top_sentences = [s[2] for s in sorted(top, key=lambda x: x[1])]

        # Key points: top 4 short sentences
        key_points = []
        for _, _, sent in sorted(scored, reverse=True):
            if len(sent.split()) <= 30 and sent not in key_points:
                key_points.append(sent.strip())
            if len(key_points) >= 4:
                break

        return {
            "summary": ' '.join(top_sentences),
            "key_points": key_points,
            "urgency": self._detect_urgency(text),
            "document_type": self._detect_doc_type(text),
            "sentence_count": len(sentences)
        }


# ═══════════════════════════════════════════════════
#  MODEL 3 — RULE-BASED COMPLETENESS CHECKER
#  Checks for 16 mandatory CDSCO SAE fields
# ═══════════════════════════════════════════════════

class CompletenessChecker:
    def __init__(self):
        # Each field: (field_id, display_name, [list of regex/keyword patterns])
        self.mandatory_fields = [
            ("report_reference",       "Report Reference",
             [r'CDSCO/\S+', r'Report\s*(Ref|Reference|No|ID)\s*:', r'SAE[-\s]?\d+']),

            ("report_date",            "Report Date",
             [r'Date\s+of\s+Report\s*:', r'Reported\s+on\s*:', r'Report\s+Date\s*:']),

            ("patient_age_sex",        "Patient Age & Sex",
             [r'\b\d{1,3}\s*(?:years?|yrs?|year-old)\b', r'\b(Male|Female|M|F)\b',
              r'Age\s*:\s*\d+']),

            ("patient_id",             "Patient ID/Initials",
             [r'Patient\s*(ID|Identifier|Initials)\s*:', r'PT-\d+',
              r'Subject\s*ID\s*:', r'Patient\s*:\s*[A-Z]']),

            ("suspect_drug",           "Suspect Drug & Dose",
             [r'Suspect\s+Drug\s*:', r'Drug\s*:\s*[A-Za-z]',
              r'\b\d+\s*mg\b', r'Dose\s*:\s*\d+']),

            ("batch_number",           "Batch / Lot Number",
             [r'Batch\s*(No|Number|#)\s*:', r'Lot\s*(No|Number)\s*:',
              r'[A-Z]{2,}\d{4,}']),

            ("manufacturer",           "Manufacturer",
             [r'Manufacturer\s*:', r'Manufactured\s+by\s*:',
              r'(Ltd|Limited|Pharma|Pharmaceuticals|Industries)\b']),

            ("ae_description",         "Adverse Event Description",
             [r'Adverse\s+Event\s*:', r'AE\s*:', r'Event\s*:',
              r'Reaction\s*:', r'complaint\s*of\b']),

            ("ae_onset_date",          "AE Onset Date",
             [r'(Onset|Start)\s+Date\s*:', r'Date\s+of\s+(Event|Onset|AE)\s*:',
              r'AE\s+Onset\s*:', r'Day\s+\d+\s+of\s+treatment']),

            ("seriousness",            "Seriousness Criteria",
             [r'Seriousness\s*:', r'Serious\s*(Adverse|Event|AE)\s*:',
              r'\b(death|disability|hospitali|life.threatening|congenital)\b']),

            ("causality",              "Causality Assessment",
             [r'Causality\s*:', r'Naranjo\s*(Score|Scale)\s*:',
              r'\b(probable|possible|definite|unlikely|unrelated|certain)\b']),

            ("outcome",                "Patient Outcome",
             [r'Outcome\s*:', r'\b(recovered|recovering|fatal|death|ongoing|sequelae|resolved)\b',
              r'Patient\s+outcome\s*:']),

            ("reporter_name",          "Reporter Name",
             [r'Reporter\s*(Name)?\s*:', r'Reported\s+by\s*:',
              r'Investigator\s*:', r'Dr\.\s+[A-Z][a-z]+']),

            ("reporter_qualification", "Reporter Qualification",
             [r'(MBBS|MD|MS|DNB|PhD|DO|FRCP|DM)\b',
              r'Qualification\s*:', r'Designation\s*:']),

            ("reporter_contact",       "Reporter Contact",
             [r'\b[6-9]\d{9}\b', r'\+91', r'@',
              r'Tel\s*:', r'Contact\s*:', r'Phone\s*:']),

            ("narrative",              "Clinical Narrative",
             [r'Narrative\s*:', r'Clinical\s+course\s*:',
              r'Patient\s+(presented|was|developed|admitted)',
              r'History\s*:', r'Summary\s*:']),
        ]

    def check(self, text):
        """Check each mandatory field and return completeness report."""
        results = {}
        present = []
        missing = []
        incomplete = []

        for field_id, display_name, patterns in self.mandatory_fields:
            found = False
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found = True
                    break
            if found:
                results[field_id] = "present"
                present.append(display_name)
            else:
                results[field_id] = "missing"
                missing.append(display_name)

        total = len(self.mandatory_fields)
        score = int((len(present) / total) * 100)

        if score >= 85:
            recommendation = "Approve for review"
        elif score >= 60:
            recommendation = "Return for completion"
        else:
            recommendation = "Reject - insufficient data"

        notes = self._build_notes(missing, score, recommendation)

        return {
            "fields": results,
            "present": present,
            "missing": missing,
            "score": score,
            "recommendation": recommendation,
            "notes": notes
        }

    def _build_notes(self, missing, score, recommendation):
        if not missing:
            return "All mandatory CDSCO fields are present. Document is complete for regulatory review."
        notes = f"Compliance score: {score}%. "
        notes += f"Recommendation: {recommendation}. "
        if missing:
            notes += f"The following {len(missing)} mandatory field(s) are missing or not clearly stated: "
            notes += ', '.join(missing[:6])
            if len(missing) > 6:
                notes += f" and {len(missing)-6} more"
            notes += ". Please request the reporter to resubmit with complete information."
        return notes


# ═══════════════════════════════════════════════════
#  MODEL 4 — TF-IDF + MULTINOMIAL NAIVE BAYES CLASSIFIER
#  Trains on SQLite database training data
#  Classifies: Death | Disability | Hospitalization | Other
# ═══════════════════════════════════════════════════

class NaiveBayesClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),       # unigrams, bigrams, trigrams
                max_features=5000,
                min_df=1,
                stop_words='english',
                sublinear_tf=True,        # log TF scaling
            )),
            ('nb', MultinomialNB(
                alpha=0.3               # Laplace smoothing
            ))
        ])
        self.label_encoder = LabelEncoder()
        self.classes = ['Death', 'Disability', 'Hospitalization', 'Other']
        self.is_trained = False

    def load_training_data(self):
        """Load training samples from SQLite database. Initializes if empty."""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT text, label FROM training_data")
            rows = c.fetchall()
            conn.close()
        except sqlite3.OperationalError:
            # Table missing - initialize
            print("[ReguAI] Database tables missing. Initialising...")
            self._init_db()
            return self.load_training_data()

        if not rows:
            # Table exists but empty - seed
            print("[ReguAI] Training data empty. Seeding...")
            self._init_db()
            return self.load_training_data()

        texts = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        return texts, labels

    def _init_db(self):
        """Create and seed database tables."""
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT, raw_text TEXT,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS anonymisation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER, anon_text TEXT, entities_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS summarisation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER, summary TEXT, key_points TEXT, urgency TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS completeness_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER, score INTEGER, missing_fields TEXT,
                present_fields TEXT, recommendation TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS classification_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER, label TEXT,
                confidence_death REAL, confidence_disability REAL,
                confidence_hospitalization REAL, confidence_other REAL,
                priority TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL, label TEXT NOT NULL
            );
        """)
        
        training_samples = [
            ("patient found unresponsive cardiac arrest death pronounced dead post-mortem fatal outcome", "Death"),
            ("patient died following severe adverse reaction fatal", "Death"),
            ("death reported acute myocardial infarction patient expired hospital mortality", "Death"),
            ("permanent disability paralysis unable to walk persistent neurological deficit", "Disability"),
            ("peripheral neuropathy permanent nerve damage disability grade 3", "Disability"),
            ("patient admitted hospital emergency hospitalization ICU admission ward", "Hospitalization"),
            ("hospitalised acute kidney injury ICU stay hospital admission required", "Hospitalization"),
            ("mild rash itching grade 1 resolved no treatment required minor adverse event", "Other"),
            ("headache dizziness grade 1 adverse event outpatient managed", "Other")
        ]
        c.executemany("INSERT INTO training_data (text, label) VALUES (?, ?)", training_samples)
        conn.commit()
        conn.close()
        print("[ReguAI] Database initialised and seeded.")

    def train(self):
        """Train TF-IDF + Naive Bayes on database samples."""
        texts, labels = self.load_training_data()
        if not texts:
            print("[WARNING] No training data found in database.")
            return False
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        print(f"[OK] Naive Bayes trained on {len(texts)} samples")
        # Cross-validation accuracy
        if len(texts) >= 5:
            scores = cross_val_score(self.pipeline, texts, labels, cv=min(5, len(texts)//2))
            print(f"[OK] Cross-validation accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
        return True

    def classify(self, text):
        """Classify text and return label + confidence scores."""
        if not self.is_trained:
            self.train()

        # Get probability for each class
        proba = self.pipeline.predict_proba([text])[0]
        class_names = self.pipeline.classes_

        # Map to our 4 labels
        conf = {c: 0.0 for c in self.classes}
        for i, cls in enumerate(class_names):
            if cls in conf:
                conf[cls] = round(float(proba[i]) * 100, 1)

        label = max(conf, key=conf.get)

        # Assign priority
        priority_map = {
            'Death': 'P1-Critical',
            'Disability': 'P2-High',
            'Hospitalization': 'P3-Medium',
            'Other': 'P4-Low'
        }

        reasoning = self._generate_reasoning(text, label, conf)

        return {
            "label": label,
            "confidence": conf,
            "priority": priority_map.get(label, 'P4-Low'),
            "reasoning": reasoning
        }

    def _generate_reasoning(self, text, label, conf):
        """Generate a rule-based reasoning explanation."""
        text_lower = text.lower()
        reasons = []

        if label == 'Death':
            triggers = [w for w in ['death','died','fatal','deceased','expired','mortality','cardiac arrest'] if w in text_lower]
            reasons.append(f"Document contains death-indicating terms: {', '.join(triggers[:3]) if triggers else 'fatal outcome pattern'}.")
        elif label == 'Disability':
            triggers = [w for w in ['disability','paralysis','permanent','unable to walk','neuropathy','residual deficit'] if w in text_lower]
            reasons.append(f"Document contains disability-indicating terms: {', '.join(triggers[:3]) if triggers else 'permanent impairment pattern'}.")
        elif label == 'Hospitalization':
            triggers = [w for w in ['admitted','hospitalised','hospitalized','icu','ward','inpatient','emergency'] if w in text_lower]
            reasons.append(f"Document contains hospitalisation-indicating terms: {', '.join(triggers[:3]) if triggers else 'admission pattern'}.")
        else:
            reasons.append("Document does not contain strong indicators for Death, Disability, or Hospitalisation.")

        reasons.append(f"Naive Bayes classifier confidence: {conf[label]:.1f}% for '{label}' class.")
        return ' '.join(reasons)


# ═══════════════════════════════════════════════════
#  DATABASE HELPER
# ═══════════════════════════════════════════════════

def save_document(text, filename="manual_input"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename, raw_text) VALUES (?, ?)", (filename, text))
    doc_id = c.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def save_results(table, doc_id, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if table == 'anonymisation_results':
        c.execute("INSERT INTO anonymisation_results (document_id, anon_text, entities_json) VALUES (?,?,?)",
                  (doc_id, data['anonymised_text'], json.dumps(data['entities'])))
    elif table == 'summarisation_results':
        c.execute("INSERT INTO summarisation_results (document_id, summary, key_points, urgency) VALUES (?,?,?,?)",
                  (doc_id, data['summary'], json.dumps(data['key_points']), data['urgency']))
    elif table == 'completeness_results':
        c.execute("INSERT INTO completeness_results (document_id, score, missing_fields, present_fields, recommendation) VALUES (?,?,?,?,?)",
                  (doc_id, data['score'], json.dumps(data['missing']), json.dumps(data['present']), data['recommendation']))
    elif table == 'classification_results':
        conf = data.get('confidence', {})
        c.execute("INSERT INTO classification_results (document_id, label, confidence_death, confidence_disability, confidence_hospitalization, confidence_other, priority) VALUES (?,?,?,?,?,?,?)",
                  (doc_id, data['label'], conf.get('Death',0), conf.get('Disability',0),
                   conf.get('Hospitalization',0), conf.get('Other',0), data['priority']))
    conn.commit()
    conn.close()

def get_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT d.id, d.filename, d.upload_time,
               cr.label, cr.priority,
               co.score
        FROM documents d
        LEFT JOIN classification_results cr ON cr.document_id = d.id
        LEFT JOIN completeness_results co ON co.document_id = d.id
        ORDER BY d.upload_time DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"id":r[0],"filename":r[1],"time":r[2],"label":r[3],"priority":r[4],"score":r[5]} for r in rows]
