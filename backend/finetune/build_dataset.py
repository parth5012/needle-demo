"""Build LoRA finetune JSONL for MAIN artisan onboarding (5 fields).

Format (per needle/model/finetune.py):
  {"query": "<narrative>", "reasoning": "<short derivation>",
   "answers": [{"name": "ArtisanNeuralProfile", "arguments": {...}}],
   "tools": [<artisan_schema>], "system": "<_NEURAL_SYSTEM>"}

Covers: Bhojpuri/Maithili/Kannada/Gujarati dialects, phone-vs-PIN,
PEH-/TRIFED- slot correctness, greeting-as-name guard, full IDs,
off-topic refusals (answers=[]).
Run: .venv/Scripts/python.exe finetune/build_dataset.py

Round 3: training queries are chunk-expanded with the SAME sentence-aware
chunker the engine uses at inference (`app.needle_engine._chunk_text`), so the
model learns on the fragments it will actually see. Each chunk keeps only the
fields evidenced verbatim in that chunk; chunks with no evidenced fields become
refusal examples (teaches no-hallucination on fragments like trailing
GI-tag sentences).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.needle_engine import _chunk_text  # same chunker as inference

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "artisan_schema.json"), encoding="utf-8") as f:
    SCHEMA = json.load(f)

SYSTEM = (
    "Extract the artisan's personal identity details from the narrative. "
    "The name is the person's full personal name, never a greeting "
    "(never namaste, namaskara, pranam, hello, vanakkam) and never a place, "
    "cluster, or craft. Copy phone numbers, PIN codes, and government IDs "
    "exactly as written."
)

BASE = [
    ("Pranam! Humar naam Gauri Devi ba. Hum Varanasi Handloom Cluster me pichhle 18 years se authentic Banarasi brocade bunat baani. Contact number 9876543210 ba aur humar security pin 1234 rakhla ba. Humaar government Pehchan ID PEH-IND-88320 ha aur TRIFED registration number TRIFED-UP-VNS-1049 ba.",
     {"name": "Gauri Devi", "phone": "9876543210", "pin": "1234", "pehchan_id": "PEH-IND-88320", "trifed_id": "TRIFED-UP-VNS-1049"},
     "Humar naam -> Gauri Devi; Contact number 10-digit -> phone; security pin 4-digit -> pin; PEH- prefix -> pehchan_id; TRIFED- prefix -> trifed_id."),
    ("Namaskara, I am Ramesh Gowda, a traditional lacquerware artisan from Channapatna Craft Cluster, Ramanagara, Karnataka. Reach me at phone: 9123456780, pin: 4321. My Pehchan card is PEH-KA-44912 and TRIFED ID is TRIFED-KA-CHN-8821.",
     {"name": "Ramesh Gowda", "phone": "9123456780", "pin": "4321", "pehchan_id": "PEH-KA-44912", "trifed_id": "TRIFED-KA-CHN-8821"},
     "I am -> Ramesh Gowda (not Namaskara greeting); phone 10-digit vs pin 4-digit; PEH- and TRIFED- by own prefix."),
    ("Pranam, I am Sunita Jha from Madhubani, Bihar. My mobile number is 9845123456, pin is 5678. Pehchan ID PEH-BR-99012 and TRIFED ID TRIFED-BR-MTH-3021.",
     {"name": "Sunita Jha", "phone": "9845123456", "pin": "5678", "pehchan_id": "PEH-BR-99012", "trifed_id": "TRIFED-BR-MTH-3021"},
     "I am -> Sunita Jha; mobile 10-digit -> phone; pin 4-digit; IDs routed by PEH-/TRIFED- prefix."),
    ("Namaste! My name is Fatima Khatri from Bhuj, Kutch district in Gujarat. Phone number: 9988776655, pin: 9876. Pehchan ID PEH-GJ-77123, TRIFED-GJ-KCH-1102.",
     {"name": "Fatima Khatri", "phone": "9988776655", "pin": "9876", "pehchan_id": "PEH-GJ-77123", "trifed_id": "TRIFED-GJ-KCH-1102"},
     "My name is -> Fatima Khatri (not Namaste); phone vs pin by length; IDs by prefix."),
    ("Vanakkam, I am Meena Krishnan from Kanchipuram. Contact 9790123456, security pin 2468. Pehchan PEH-TN-55123, TRIFED TRIFED-TN-KPM-2201.",
     {"name": "Meena Krishnan", "phone": "9790123456", "pin": "2468", "pehchan_id": "PEH-TN-55123", "trifed_id": "TRIFED-TN-KPM-2201"},
     "Vanakkam is greeting not name; phone 10-digit starting 9; pin 4-digit; IDs by prefix."),
    ("Kem cho! Hun naam Arjun Rathod chhe, Morbi cluster thi. Phone 9898012345, pin 1122. Pehchan ID PEH-GJ-90210, TRIFED ID TRIFED-GJ-MRB-3301.",
     {"name": "Arjun Rathod", "phone": "9898012345", "pin": "1122", "pehchan_id": "PEH-GJ-90210", "trifed_id": "TRIFED-GJ-MRB-3301"},
     "Hun naam -> Arjun Rathod (not Kem cho); phone vs pin; IDs by prefix."),
    ("Ram ram! Humaar naam Dinesh Kumar ba, Lucknow Chikankari cluster se. Mobile 9005012345, pin 7788. Pehchan PEH-UP-11023, TRIFED TRIFED-UP-LKO-4410.",
     {"name": "Dinesh Kumar", "phone": "9005012345", "pin": "7788", "pehchan_id": "PEH-UP-11023", "trifed_id": "TRIFED-UP-LKO-4410"},
     "Humaar naam -> Dinesh Kumar (hum- possessive not name); phone 10-digit; pin 4-digit."),
    ("Hello, I am Parth, contact me at 9038749127, pehchan id PEH-GJ-71231.",
     {"name": "Parth", "phone": "9038749127", "pehchan_id": "PEH-GJ-71231"},
     "Single-word name Parth; comma-delimited phone and PEH- ID; no pin/trifed present so omitted."),
    ("Nomoshkar, ami Ananya Das, Bishnupur Baluchari cluster theke. Phone 9830012345, pin 6420. Pehchan ID PEH-WB-81231, TRIFED ID TRIFED-WB-BSH-5511.",
     {"name": "Ananya Das", "phone": "9830012345", "pin": "6420", "pehchan_id": "PEH-WB-81231", "trifed_id": "TRIFED-WB-BSH-5511"},
     "Nomoshkar is a Bengali greeting, never the name; three tokens Ananya Das all kept."),
    ("Namaskara, I am Ramesh Kumar Gowda from Channapatna Craft Cluster. Phone 9123456780, pin 4321, Pehchan PEH-KA-44912.",
     {"name": "Ramesh Kumar Gowda", "phone": "9123456780", "pin": "4321", "pehchan_id": "PEH-KA-44912"},
     "Three-part name kept whole; never truncate to two tokens."),
    ("I am Ravi Kumar from Patna. My security pin is 2468 and my contact number is 9771012345. Pehchan PEH-BH-30121, TRIFED TRIFED-BH-PAT-1121.",
     {"name": "Ravi Kumar", "phone": "9771012345", "pin": "2468", "pehchan_id": "PEH-BH-30121", "trifed_id": "TRIFED-BH-PAT-1121"},
     "PIN mentioned before phone: route by digit length (4 vs 10), never by order."),
    ("Namaste, Meera Nair here, pehchan PEH-KL-88121, contact 9847012346.",
     {"name": "Meera Nair", "phone": "9847012346", "pehchan_id": "PEH-KL-88121"},
     "Terse phrasing with 'here': name Meera Nair; pehchan and contact both captured."),
    ("Good morning, my name is Anil Verma. Phone +91 9811023456. PIN-3344. Govt Pehchan: PEH-DL-20314. TRIFED registration: TRIFED-DL-0091.",
     {"name": "Anil Verma", "phone": "9811023456", "pin": "3344", "pehchan_id": "PEH-DL-20314", "trifed_id": "TRIFED-DL-0091"},
     "+91 prefix stripped to 10-digit phone; PIN-3344 normalized to 3344; IDs by prefix."),
    ("Pranam! Humar naam Bablu Paswan ba. Sampark number 8765123450, suraksha pin 9090. Pehchan ID PEH-BR-55110, TRIFED number TRIFED-BR-PAT-7712.",
     {"name": "Bablu Paswan", "phone": "8765123450", "pin": "9090", "pehchan_id": "PEH-BR-55110", "trifed_id": "TRIFED-BR-PAT-7712"},
     "Dialect possessives humar/humaar ignored; 10-digit vs 4-digit; slot by prefix."),
]

REFUSALS = [
    "Hello I make crafts.",
    "What is the weather in Jaipur today?",
    "Tell me about GI tags for Banarasi sarees.",
]

# Round 2 (post-training diagnosis): the full-length UI preset (~470 chars,
# two IDs in one dialect sentence) made the tuned model file PEH- under the
# TRIFED value. These long two-ID examples teach per-prefix routing in situ.
LONG_TWO_ID = [
    ("Pranam! Humar naam Gauri Devi ba. Hum Varanasi Handloom Cluster me pichhle 18 years se authentic Banarasi brocade aur pure silk saree bunat baani. Contact number 9876543210 ba aur humar security pin 1234 rakhla ba. Humaar government Pehchan ID PEH-IND-88320 ha aur TRIFED registration number TRIFED-UP-VNS-1049 ba. Hum Banarasi Brocade GI tagged saari banawat baani.",
     {"name": "Gauri Devi", "phone": "9876543210", "pin": "1234", "pehchan_id": "PEH-IND-88320", "trifed_id": "TRIFED-UP-VNS-1049"},
     "Full-length preset: PEH-IND-88320 -> pehchan_id and TRIFED-UP-VNS-1049 -> trifed_id, each routed by its own prefix even sharing one sentence."),
    ("Pranam! Humar naam Bablu Paswan ba, Varanasi me Banarasi saree bunat baani pichhle 22 years se. Contact number 8765123450 ba aur security pin 9090 ba. Sarkari Pehchan ID PEH-UP-77110 ha aur TRIFED registration TRIFED-UP-VNS-8821 ba. Hum GI tagged craft banawat baani.",
     {"name": "Bablu Paswan", "phone": "8765123450", "pin": "9090", "pehchan_id": "PEH-UP-77110", "trifed_id": "TRIFED-UP-VNS-8821"},
     "Long dialect narrative: two IDs one sentence; PEH- -> pehchan_id, TRIFED- -> trifed_id; 22 years is experience, never phone."),
    ("Namaste! Humaar naam Rekha Devi ba, Bagru Hand Block Printing Cluster, Jaipur se. Contact 9785012345, suraksha pin 8899. TRIFED registration TRIFED-RJ-JPR-2204 ha aur Pehchan ID PEH-RJ-78120 ba (TRIFED mentioned first).",
     {"name": "Rekha Devi", "phone": "9785012345", "pin": "8899", "pehchan_id": "PEH-RJ-78120", "trifed_id": "TRIFED-RJ-JPR-2204"},
     "Reversed ID order: route strictly by PEH-/TRIFED- prefix, never by mention order."),
    ("Hello, my name is Kavita Singh from Jaipur Blue Pottery cluster with 15 years of experience. You can reach me on 9829012345 and my security pin is 4455. My TRIFED registration is TRIFED-RJ-JPR-1188 and my Pehchan card is PEH-RJ-31230.",
     {"name": "Kavita Singh", "phone": "9829012345", "pin": "4455", "pehchan_id": "PEH-RJ-31230", "trifed_id": "TRIFED-RJ-JPR-1188"},
     "English long form with 15 years experience (not phone); TRIFED-first order still routes by prefix."),
]

VARIANTS = [
    ("My name is {name} from {place}. Phone {phone}, pin {pin}. Pehchan {peh}, TRIFED {tri}.", "Canonical English template."),
    ("Pranam! Humar naam {name} ba. Contact number {phone} ba aur security pin {pin} ba. Pehchan ID {peh} ha aur TRIFED number {tri} ba.", "Bhojpuri template with possessives."),
    ("Namaskara, I am {name}, {place} artisan. Reach me at {phone}, pin {pin}. Pehchan card {peh}, TRIFED ID {tri}.", "Greeting-led template; name never the greeting."),
    ("{name} | {phone} | {pin} | {peh} | {tri}", "Pipe-delimited terse template."),
]

PEOPLE = [
    ("Kavita Singh", "Jaipur Blue Pottery cluster", "9829012345", "4455", "PEH-RJ-31230", "TRIFED-RJ-JPR-1188"),
    ("Mohan Lal", "Moradabad Brassware Cluster", "9412012345", "6677", "PEH-UP-40981", "TRIFED-UP-MBD-5520"),
    ("Lakshmi Amma", "Kondapalli Bommallu Cluster", "9490123456", "2233", "PEH-AP-67120", "TRIFED-AP-KRI-9031"),
    ("Rekha Devi", "Bagru Hand Block Printing Cluster", "9785012345", "8899", "PEH-RJ-78120", "TRIFED-RJ-JPR-2204"),
    ("Suresh Prajapati", "Khurja Pottery Cluster", "9837012345", "1010", "PEH-UP-55671", "TRIFED-UP-KHJ-3312"),
    ("Asha Ben", "Patan Patola Cluster", "9879112345", "2020", "PEH-GJ-34451", "TRIFED-GJ-PTN-6710"),
]


def chunk_expand(query, args, reasoning):
    """Split a (query, args) pair the way inference sees it.

    Returns one training row per chunk: only fields evidenced verbatim in that
    chunk are kept; field-less chunks become refusal rows (answers=[]).
    """
    rows = []
    for ch in _chunk_text(query):
        kept = {k: v for k, v in args.items()
                if v is not None and str(v) in ch}
        if kept:
            rows.append((ch,
                         f"Chunk-scoped: {reasoning} Kept only fields evidenced in this chunk: {sorted(kept)}.",
                         kept))
        else:
            rows.append((ch,
                         "Chunk-scoped: no name/phone/PIN/IDs evidenced in this chunk fragment, so no call.",
                         {}))
    return rows


def emit(rows, query, args, reasoning):
    # Full-text row: teaches cross-sentence routing (e.g. two IDs in one
    # dialect sentence — the signal round 2 learned and round 3 lost).
    rows.append({"query": query, "reasoning": reasoning,
                 "answers": ([{"name": "ArtisanNeuralProfile", "arguments": args}]
                             if args else []),
                 "tools": [SCHEMA], "system": SYSTEM})
    # Chunk rows: same distribution inference sees (round 3).
    if args:
        for ch, r, kept in chunk_expand(query, args, reasoning):
            rows.append({"query": ch, "reasoning": r,
                         "answers": ([{"name": "ArtisanNeuralProfile", "arguments": kept}]
                                     if kept else []),
                         "tools": [SCHEMA], "system": SYSTEM})


def main():
    rows = []
    for q, args, reasoning in BASE + LONG_TWO_ID:
        emit(rows, q, args, reasoning)
    for tpl, note in VARIANTS:
        for name, place, phone, pin, peh, tri in PEOPLE:
            q = tpl.format(name=name, place=place, phone=phone, pin=pin, peh=peh, tri=tri)
            emit(rows, q,
                 {"name": name, "phone": phone, "pin": pin,
                  "pehchan_id": peh, "trifed_id": tri},
                 f"{note} name={name}; phone {phone} (10-digit) vs pin {pin} (4-digit); {peh} -> pehchan_id; {tri} -> trifed_id.")
    for q in REFUSALS:
        rows.append({"query": q, "reasoning": "No evidenced name/phone/IDs; off-topic so no call.",
                     "answers": [], "tools": [SCHEMA], "system": SYSTEM})
    # partial-field examples (missing pin/trifed stay omitted)
    emit(rows, "I am Vikram, phone 9818012345.",
         {"name": "Vikram", "phone": "9818012345"},
         "Name Vikram; 10-digit phone; no pin or IDs evidenced so omitted.")
    emit(rows, "Namaste, Meera Nair here, pehchan PEH-KL-88120, contact 9847012345.",
         {"name": "Meera Nair", "phone": "9847012345", "pehchan_id": "PEH-KL-88120"},
         "Namaste is greeting; name Meera Nair; PEH- -> pehchan_id; 10-digit -> phone.")
    out = os.path.join(HERE, "artisan_main.jsonl")
    with open(out, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples  {out}")


if __name__ == "__main__":
    main()
