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

Round 4: LONG_TWO_ID gains dialect + terse two-ID rows (Bhojpuri/Maithili/
Telugu/Bengali/Punjabi/Nagpuri/Marwari/Kannada sentences sharing both IDs,
unlabeled comma pairs, jam-clauses, no-TRIFED variants) targeting the round-3
failure mode: pehchan_id dropped (74%) when a second ID follows in the same
sentence or list. Every row's reasoning repeats: BOTH IDs in ONE call.
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
#
# Round 4 (post-training diagnosis, round-3 weights @ 23/35): pehchan_id is
# the dropped field (74%) whenever two IDs share one dialect sentence or a
# terse comma list — the model emits TRIFED- but skips PEH-. These rows force
# a single tool call that carries BOTH IDs (plus jam-clause and no-trifed
# variants), routed by prefix, nothing dropped.
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
    # --- Round 4: dialect two-ID sentences (pehchan first, then trifed) ---
    ("Pranam! Humar naam Shivani Devi ba, Bhagalpur silk cluster se. Contact 9431212345, pin 5566. Sarkari Pehchan ID PEH-BR-44120 ha aur TRIFED registration TRIFED-BR-BGP-6610 ba.",
     {"name": "Shivani Devi", "phone": "9431212345", "pin": "5566", "pehchan_id": "PEH-BR-44120", "trifed_id": "TRIFED-BR-BGP-6610"},
     "Bhojpuri two-ID sentence: PEH-BR-44120 -> pehchan_id AND TRIFED-BR-BGP-6610 -> trifed_id, BOTH in one call; never drop the PEH- value when a TRIFED- value follows it."),
    ("Pranam, humar naam Priya Mishra achhi, Darbhanga, Bihar se. Mithila painting cluster me 12 years se kaaj ba. Mobile 9431412345, pin 3434. Pehchan ID PEH-BR-62130 ha aur TRIFED number TRIFED-BR-DBR-7715. Hamar kala Madhubani Paintings GI tag laabhla achhi.",
     {"name": "Priya Mishra", "phone": "9431412345", "pin": "3434", "pehchan_id": "PEH-BR-62130", "trifed_id": "TRIFED-BR-DBR-7715"},
     "Maithili long narrative with GI tail: both IDs share one sentence — PEH-BR-62130 -> pehchan_id and TRIFED-BR-DBR-7715 -> trifed_id in the same call."),
    ("Namaste, nenu Ravi Teja, Kondapalli Bommallu Cluster nundi. Phone 9491112345, pin 4545. Pehchan ID PEH-AP-73140 mariyu TRIFED ID TRIFED-AP-VJA-8825.",
     {"name": "Ravi Teja", "phone": "9491112345", "pin": "4545", "pehchan_id": "PEH-AP-73140", "trifed_id": "TRIFED-AP-VJA-8825"},
     "Telugu narrative: mariyu-joined two-ID sentence; emit both PEH-AP-73140 and TRIFED-AP-VJA-8825 in one call, routed by prefix."),
    ("Nomoshkar, ami Arpita Sen, Bishnupur Baluchari cluster theke. Phone 9830212345, pin 5656. Pehchan ID PEH-WB-92150 ha aur TRIFED ID TRIFED-WB-BSH-6635.",
     {"name": "Arpita Sen", "phone": "9830212345", "pin": "5656", "pehchan_id": "PEH-WB-92150", "trifed_id": "TRIFED-WB-BSH-6635"},
     "Bengali narrative: two IDs one sentence; PEH-WB-92150 -> pehchan_id and TRIFED-WB-BSH-6635 -> trifed_id together, never one without the other."),
    ("Sat Sri Akal! Mera naam Gurpreet Singh hai, Amritsar Phulkari cluster ton. Phone 9872112345, pin 6767. Pehchan ID PEH-PB-83160 hai ate TRIFED ID TRIFED-PB-ASR-7745.",
     {"name": "Gurpreet Singh", "phone": "9872112345", "pin": "6767", "pehchan_id": "PEH-PB-83160", "trifed_id": "TRIFED-PB-ASR-7745"},
     "Punjabi narrative: ate-joined two-ID sentence; both PEH-PB-83160 and TRIFED-PB-ASR-7745 in one call by prefix."),
    ("Johar! Humra naam Munna Lal hai, Ranchi Dokra cluster se. Sampark 9771412345, pin 5757. Pehchan PEH-JH-73170 ha aur TRIFED TRIFED-JH-RNC-8855.",
     {"name": "Munna Lal", "phone": "9771412345", "pin": "5757", "pehchan_id": "PEH-JH-73170", "trifed_id": "TRIFED-JH-RNC-8855"},
     "Nagpuri narrative: unlabeled Pehchan/TRIFED pair in one sentence; emit BOTH IDs, PEH-JH-73170 and TRIFED-JH-RNC-8855."),
    ("Ram ram sa! Mharo naam Bhavna Joshi hai, Jodhpur wooden handicraft cluster se. Phone 9825112345, pin 8686. Pehchan ID PEH-RJ-94180 chhe ane TRIFED registration TRIFED-RJ-JDH-6645.",
     {"name": "Bhavna Joshi", "phone": "9825112345", "pin": "8686", "pehchan_id": "PEH-RJ-94180", "trifed_id": "TRIFED-RJ-JDH-6645"},
     "Marwari narrative: two IDs one sentence; PEH-RJ-94180 -> pehchan_id and TRIFED-RJ-JDH-6645 -> trifed_id, both in the same call."),
    ("Namaskara, I am Girish Rao from Channapatna. Phone 9886112345, pin 1919. Pehchan ID PEH-KA-55190 mattu TRIFED ID TRIFED-KA-CHN-9965.",
     {"name": "Girish Rao", "phone": "9886112345", "pin": "1919", "pehchan_id": "PEH-KA-55190", "trifed_id": "TRIFED-KA-CHN-9965"},
     "Kannada narrative: mattu-joined two-ID sentence; emit both PEH-KA-55190 and TRIFED-KA-CHN-9965 together."),
    # --- Round 4: terse/jam shapes (the exact eval failing forms) ---
    ("Kavita Singh here, Jaipur Blue Pottery. Contact 9829012345, pin 4455. Pehchan PEH-RJ-31230, TRIFED TRIFED-RJ-JPR-1188.",
     {"name": "Kavita Singh", "phone": "9829012345", "pin": "4455", "pehchan_id": "PEH-RJ-31230", "trifed_id": "TRIFED-RJ-JPR-1188"},
     "Terse comma-separated ID pair with no 'ID' word: PEH-RJ-31230 -> pehchan_id, TRIFED-RJ-JPR-1188 -> trifed_id; both must appear even when unlabeled."),
    ("Lakshmi Amma, Kondapalli cluster, phone 9490123456 pin 2233, IDs PEH-AP-67120 and TRIFED-AP-KRI-9031 for verification.",
     {"name": "Lakshmi Amma", "phone": "9490123456", "pin": "2233", "pehchan_id": "PEH-AP-67120", "trifed_id": "TRIFED-AP-KRI-9031"},
     "Everything jammed in one clause: one call carries name, phone, pin, pehchan_id AND trifed_id — nothing dropped."),
    ("Rekha Devi, Bagru cluster Jaipur. Phone 9785012345, pin 8899. TRIFED TRIFED-RJ-JPR-2204, Pehchan PEH-RJ-78120.",
     {"name": "Rekha Devi", "phone": "9785012345", "pin": "8899", "pehchan_id": "PEH-RJ-78120", "trifed_id": "TRIFED-RJ-JPR-2204"},
     "TRIFED mentioned first in a terse comma pair; still route by prefix — PEH-RJ-78120 -> pehchan_id even though it comes last."),
    ("Suresh Prajapati, Khurja Pottery. Sampark 9837012345, pin 1010 aur Pehchan PEH-UP-55671 ha.",
     {"name": "Suresh Prajapati", "phone": "9837012345", "pin": "1010", "pehchan_id": "PEH-UP-55671"},
     "No TRIFED present: pehchan PEH-UP-55671 must still be emitted after pin; missing fields stay absent, present fields never dropped."),
    ("Mohan Lal, Moradabad brass cluster. Phone 9412012345, pin 6677, Pehchan PEH-UP-40981, TRIFED TRIFED-UP-MBD-5520 ek line me.",
     {"name": "Mohan Lal", "phone": "9412012345", "pin": "6677", "pehchan_id": "PEH-UP-40981", "trifed_id": "TRIFED-UP-MBD-5520"},
     "Four data fields in one clause: single call with phone, pin, pehchan_id and trifed_id together — prefix routing, nothing dropped."),
    ("Asha Ben, Patan Patola cluster. Contact number 9879112345, pin no. 2020, Pehchan card PEH-GJ-34451 aur TRIFED number TRIFED-GJ-PTN-6710.",
     {"name": "Asha Ben", "phone": "9879112345", "pin": "2020", "pehchan_id": "PEH-GJ-34451", "trifed_id": "TRIFED-GJ-PTN-6710"},
     "'Pehchan card ... aur TRIFED number ...' in one sentence: emit PEH-GJ-34451 and TRIFED-GJ-PTN-6710 together."),
]

VARIANTS = [
    ("My name is {name} from {place}. Phone {phone}, pin {pin}. Pehchan {peh}, TRIFED {tri}.", "Canonical English template."),
    ("Pranam! Humar naam {name} ba. Contact number {phone} ba aur security pin {pin} ba. Pehchan ID {peh} ha aur TRIFED number {tri} ba.", "Bhojpuri template with possessives."),
    ("Namaskara, I am {name}, {place} artisan. Reach me at {phone}, pin {pin}. Pehchan card {peh}, TRIFED ID {tri}.", "Greeting-led template; name never the greeting."),
    ("{name} | {phone} | {pin} | {peh} | {tri}", "Pipe-delimited terse template."),
    ("{name}, {place}. Phone {phone}, pin {pin}. TRIFED registration {tri} and Pehchan ID {peh}.", "TRIFED mentioned before Pehchan; prefix routing."),
    ("Ram ram! Humaar naam {name} ba, {place} se. Mobile {phone}, pin {pin}. Pehchan {peh}, TRIFED {tri}.", "Awadhi dialect template."),
    ("Humaar naam {name} ba, {place} se. Mobile: {phone}, PIN {pin}. Pehchan {peh}, TRIFED {tri}.", "Mobile label with PIN template."),
    ("Namaste, I am {name} from {place}. Contact number {phone}, pin no. {pin}. Pehchan {peh}, TRIFED {tri}.", "Pin no. phrasing template."),
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
    emit(rows, "Contact 9437012345 for Pipapli applique orders, pehchan PEH-OD-51230.",
         {"phone": "9437012345", "pehchan_id": "PEH-OD-51230"},
         "No name present in text; contact -> phone; PEH- -> pehchan_id.")
    out = os.path.join(HERE, "artisan_main.jsonl")
    with open(out, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples  {out}")


if __name__ == "__main__":
    main()
