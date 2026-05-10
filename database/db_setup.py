"""
database/db_setup.py
Creates and seeds the SQLite database for ReguAI.
Run once: python database/db_setup.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "reguai.db")

def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── Table 1: Documents uploaded ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        filename    TEXT,
        raw_text    TEXT,
        upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")

    # ── Table 2: Anonymisation results ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS anonymisation_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id   INTEGER,
        anon_text     TEXT,
        entities_json TEXT,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )""")

    # ── Table 3: Summarisation results ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS summarisation_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        summary     TEXT,
        key_points  TEXT,
        urgency     TEXT,
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )""")

    # ── Table 4: Completeness check results ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS completeness_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id     INTEGER,
        score           INTEGER,
        missing_fields  TEXT,
        present_fields  TEXT,
        recommendation  TEXT,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )""")

    # ── Table 5: Classification results ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS classification_results (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id    INTEGER,
        label          TEXT,
        confidence_death          REAL,
        confidence_disability     REAL,
        confidence_hospitalization REAL,
        confidence_other          REAL,
        priority       TEXT,
        created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )""")

    # ── Table 6: Training data for Naive Bayes classifier ──
    c.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        text    TEXT NOT NULL,
        label   TEXT NOT NULL
    )""")

    # ── Seed training data ──
    training_samples = [
        # DEATH cases
        ("patient found unresponsive cardiac arrest death pronounced dead post-mortem fatal outcome brainstem herniation", "Death"),
        ("patient died following severe adverse reaction death occurred respiratory failure fatal", "Death"),
        ("death reported acute myocardial infarction patient expired hospital mortality fatal outcome", "Death"),
        ("patient pronounced dead on arrival CPR unsuccessful fatal adverse event death", "Death"),
        ("terminal outcome patient succumbed complications death certificate issued fatal", "Death"),
        ("patient passed away intracranial haemorrhage brain death fatal drug reaction", "Death"),
        ("sudden death cardiac arrhythmia patient found dead home fatal outcome", "Death"),
        ("death due to anaphylaxis airway obstruction patient expired emergency", "Death"),
        ("fatal hepatic failure liver failure death patient did not survive", "Death"),
        ("patient mortality sepsis multiorgan failure death ICU death confirmed", "Death"),

        # DISABILITY cases
        ("permanent disability paralysis unable to walk persistent neurological deficit disability", "Disability"),
        ("peripheral neuropathy permanent nerve damage disability grade 3 residual deficit", "Disability"),
        ("patient unable to perform daily activities permanent disability limb weakness", "Disability"),
        ("disability permanent cognitive impairment memory loss unable to work disability", "Disability"),
        ("permanent vision loss blindness disability severe disability grade 4", "Disability"),
        ("stroke permanent hemiplegia disability residual paralysis unable to ambulate", "Disability"),
        ("permanent hearing loss disability unable to work long term disability", "Disability"),
        ("spinal cord injury permanent disability wheelchair bound residual deficit", "Disability"),
        ("severe peripheral neuropathy permanent disability unable to walk grade 3 CTCAE", "Disability"),
        ("permanent renal failure dialysis dependency disability long-term impairment", "Disability"),

        # HOSPITALIZATION cases
        ("patient admitted hospital emergency hospitalization ICU admission ward admitted", "Hospitalization"),
        ("hospitalised acute kidney injury ICU stay hospital admission required", "Hospitalization"),
        ("emergency admission hospital required inpatient care hospitalization", "Hospitalization"),
        ("patient admitted ward observation hospitalization serious adverse event", "Hospitalization"),
        ("hospital admission required anaphylaxis treatment stabilized discharged", "Hospitalization"),
        ("ICU admission sepsis inpatient hospitalization required medical care", "Hospitalization"),
        ("patient hospitalised lactic acidosis admitted stabilized discharged", "Hospitalization"),
        ("acute pancreatitis hospitalization required admitted gastroenterology ward", "Hospitalization"),
        ("severe hypoglycemia hospitalization loss of consciousness admitted", "Hospitalization"),
        ("pneumonia hospital admission required oxygen therapy discharged recovered", "Hospitalization"),

        # OTHER cases
        ("mild rash itching grade 1 resolved no treatment required minor adverse event", "Other"),
        ("headache dizziness grade 1 adverse event outpatient managed no hospitalization", "Other"),
        ("nausea vomiting grade 2 managed outpatient no admission required", "Other"),
        ("injection site reaction mild swelling no intervention needed minor", "Other"),
        ("fatigue mild transient no significant adverse event grade 1", "Other"),
        ("mild fever temperature elevation resolved antipyretics outpatient", "Other"),
        ("constipation mild adverse event grade 1 managed dietary modification", "Other"),
        ("mild elevated liver enzymes grade 1 monitoring no treatment required", "Other"),
        ("skin irritation topical mild grade 1 no serious adverse event", "Other"),
        ("insomnia mild adverse event managed no hospitalization grade 1", "Other"),
    ]

    c.executemany("INSERT INTO training_data (text, label) VALUES (?, ?)", training_samples)
    conn.commit()
    conn.close()
    print(f"[OK] Database created at: {DB_PATH}")
    print(f"[OK] {len(training_samples)} training samples seeded")

if __name__ == "__main__":
    create_database()
