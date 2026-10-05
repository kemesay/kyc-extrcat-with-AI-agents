# ID Document Extraction API v5.0

Multi-channel identity document extraction — architecture ported from
[ID-Document-Extractor](/home/mesay/Desktop/ID-Document-Extractor), with
Gemini replaced by local Ollama vision models throughout. Built for:
**MacBook Pro (Mac17,9) | Apple M5 Pro (15-core CPU: 5 Super + 10 Performance) | 24 GB unified memory | macOS, arm64 | Ollama on Metal**

---

## Architecture: 5 channels, most-trustworthy first

A photo goes through a waterfall, not a single model call. Each channel
only runs if the ones before it didn't produce enough, and a higher
channel's field is **never overwritten** by a lower one — disagreement
sets `human_review_required` instead of silently picking a value.

```
1. Barcode / QR    PDF417 (AAMVA US/CA driver's licenses & ID cards),
                   Ethiopian Digital ID FAN (Code128) + QR
                   → CRC/Reed-Solomon verified — fully trusted on decode

2. MRZ             ICAO 9303 TD1 (national ID) / TD2 / TD3 (passport)
                   → real check-digit validation

3. AI-vision       Ollama qwen2.5vl:7b, single-shot — enriches gaps a
   crosscheck      verified source left (place of birth, issuing
                   authority, ...), never overwrites what's verified

4. AI-vision       Ollama's 4-stage sequential pipeline (moondream →
   universal        qwen2.5vl:7b → llava:13b → qwen2.5vl:7b, each stage
   fallback         sees and corrects the previous one) — full read when
                   nothing structural matched at all

5. Generic OCR     Label-alias regex sweep, multi-language ontology —
   floor            the floor. Always runs if nothing else did; never
                   returns an empty profile.
```

See `idx/pipeline.py` for the orchestrator and `CLAUDE.md`/`docs/` in
the source project for the full design philosophy this inherits.

## Local OCR engine

`idx/ocr_engine.py`: **RapidOCR** (ONNX, CPU-only) primary, tesseract
fallback/vote. Deliberately CPU-only — it never competes with Ollama for
the GPU, whose share of the 24 GB unified memory is the tightest resource
on this machine (see "Known issues" below). Most documents with a barcode or MRZ resolve via channels 1-2
without ever touching Ollama or the GPU at all.

## Ollama models (channels 3-4 only)

```
qwen2.5vl:7b      6.0GB   Field extractor + layout analyzer (multilingual)
                          Also the single-shot model for crosscheck calls
llava:13b         8.0GB   Accuracy verifier
moondream:latest  1.7GB   Fast pre-scanner
```

## Multi-country / multi-language coverage

- `idx/countries.py` + `idx/ontology/icao_prado_entities.yaml` — ISO 3166-1 / ICAO PRADO issuing-state and nationality codes.
- `idx/ontology.py` + `idx/concepts/` — per-field label aliases in 9 languages (`am`, `ar`, `de`, `en`, `es`, `fr`, `it`, `ja`, `zh`), fuzzy-matched with a same-field-only guard.
- `idx/ontology/aamva_jurisdictions.yaml` — per-US-state/Canada-province facts as data, never `if state == "..."` branches in Python.
- `idx/ethiopia_id.py` — dedicated module for the Ethiopian Digital ID ("Fayda") card, both generations (FAN and FCN), dual-calendar (Ethiopian/Gregorian) date handling.

**Known, honest gap**: AAMVA's front-only OCR fallback (reading a driver's
license visually when its barcode doesn't decode) is not implemented —
the source project's version was tuned against specific real card photos
that don't exist here; see `idx/aamva.py`'s module docstring. A driver's
license with an undecodable barcode falls through to the generic OCR/AI
tiers instead of a bespoke reader.

---

## Setup

```bash
# Create/activate the conda env (Python 3.12) and install dependencies
conda create -n myenv python=3.12   # first time only
conda activate myenv
pip install -r requirements.txt

# tesseract binary must be installed system-wide too (OCR fallback):
#   conda (any OS, no sudo): conda install -n myenv -c conda-forge tesseract
#   macOS with Homebrew:     brew install tesseract
#   Ubuntu:                  sudo apt install tesseract-ocr

# Verify Ollama models are installed
ollama list

# Universal engine database (once): local PostgreSQL + pgvector, no Docker/Homebrew needed
python -m universal.localdb setup

# Start API
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info, architecture, supported document types/languages |
| GET | `/health` | Ollama model status + local OCR engine + GPU info |
| GET | `/agents` | The 5 channels, in precedence order, with what each covers |
| POST | `/extract/id?doc_type=auto` | Async extraction (`wait=true` to block, `callback_url=` for production) |
| GET | `/extract/status/{job_id}` | Poll result |
| POST | `/extract/id/sync?doc_type=auto` | Sync extraction |
| DELETE | `/extract/job/{job_id}` | Remove job |

`doc_type`: `passport` \| `national_id` \| `drivers_license` \| `residence_permit` \| `auto` (default — tries passport, then national_id, then drivers_license, in that order). Pick `residence_permit` explicitly; its MRZ format can't be told apart from `national_id`'s.

`file_back`: optional second image (multipart field) for two-sided documents — AAMVA's PDF417 is usually on the back, the Ethiopian ID's QR is on the back and its FAN barcode is on the front. Order doesn't matter, both sides are classified by content.

---

## Universal Engine (`/universal/*`) — any country, script or ID type

A second, additive engine in [`universal/`](universal/). The original `/extract/*` pipeline above is untouched; the only change to existing code is two lines in `main.py` that mount the router. Prompt text and model assignment live in [`universal_id_extraction_prompts.py`](universal_id_extraction_prompts.py), the single source of truth.

> **Which endpoint?** `/extract/*` is the original idx pipeline (barcode → MRZ → one AI read, a few seconds). It has none of the checks below and can return a mis-parsed "MRZ" from ordinary card text. For KYC, use **`/universal/*`**.

📐 **Diagrams:** [docs/UNIVERSAL_ARCHITECTURE.md](docs/UNIVERSAL_ARCHITECTURE.md): how the agents communicate, how the token limit of small models is solved, how labels, dates and countries are decided, and how the engine is served at high traffic.

### Design principles

| Principle | What it means in practice |
|---|---|
| **The models read, the code decides** | Vision models transcribe and propose; deterministic code interprets (name order, date conventions, calendars, which number is which) and verifies against evidence that doesn't come from a language model: check digits, barcodes, OCR, calendar arithmetic. |
| **Printed only** | A value is reported only if it is printed on *this* document or machine-verified. Nationality is never inferred from the country, a city or a face; a card's title is never a name; a phone number is never a personal number. |
| **The label decides the field** | A value printed beside its label ("Date Issued", "የተሰጠበት ቀን", "Guyyaa Kenname", "签发日期") is that field, in any of ~40 languages, whatever an agent called it. |
| **Doubt goes to a person, not to a guess** | When the evidence is ambiguous (an unmarked date on an Ethiopian-calendar card, two calendar halves that disagree, an agent number OCR reads differently), the field is left empty or kept, and `review_reasons` states both readings. |
| **Quality first, speed second** | Every document gets the full agent team. Speed comes from the serving architecture, never from fewer agents. |
| **Nothing is silent** | Every replaced value stays in `alternatives`, and every change is explained in `warnings` and `flags`. |

### The pipeline at a glance

```
barcode / QR (zxing) ─────────────┐  verified barcode data is fed to the extractor as ground truth
                                  ▼
moondream → qwen2.5vl → [transcriber] → llava:13b → qwen2.5vl → qwen2.5 supervisor
 scanner    extractor    raw text        verifier    layout      (text only)
      ║  in parallel on CPU: RapidOCR + tesseract text, and RapidOCR text BOXES (the card's layout)
                                  ▼
consensus (majority · specialist · supervisor pick only if an agent read it)
                                  ▼
the card's own evidence: country it names · title it prints · value beside each label (~40 languages)
                                  ▼
interpretation: names · dates (calendar pairs, month names, EC marks) · document number · sex
                                  ▼
verification: MRZ check digits · national-ID checksums · OCR grounding · printed-only rules
              · Ge'ez ↔ Latin spelling · calendars · plausibility · identity signals
                                  ▼
per field: value · confidence · source · verified_by · flags · alternatives  +  review reasons
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/universal/info` | Agents + model status, checksum schemes, calendars, OCR engines, serving policy |
| POST | `/universal/extract` | Queued (`wait=true` to block, `callback_url=` webhook, `priority=`) |
| POST | `/universal/extract/sync` | Sync (through the same queue) |
| GET | `/universal/status/{job_id}` | Poll (shows `queue_position` while queued) |
| DELETE | `/universal/job/{job_id}` | Cancel a queued job / delete a finished one |
| GET | `/universal/health` | Readiness: hosts reachable, team models resident on the GPU, queue, database, workers |
| GET | `/universal/metrics` | Prometheus metrics (speed and quality side by side) |
| POST | `/universal/warmup` | Load every team model on every host now |

- `mode=full` (default, and by quality policy the only mode): the 5-agent team plus the transcriber. `mode=fast` returns 400 unless the server sets `UNIVERSAL_ALLOW_FAST_MODE=1`.
- `priority`: `live` (customer waiting) · `online` (default) · `batch` · `bulk` — see [Serving at high traffic](#serving-at-high-traffic).
- `doc_type` is only a hint; the card's printed title decides. Accepted: `auto`, `passport`, `national_id`, `drivers_license`, `residence_permit`, `visa`, `refugee_travel_document`, `military_id`, `employee_id`, `student_id`, `voter_id`, `work_permit`, `health_insurance_card`, `tax_id`, `other`.
- `file_back`: the back side of a two-sided card (order doesn't matter). Send both sides: dates, phone, blood group and emergency contact are often on the back.
- `reject_poor_capture=true`: refuse a blurred / dark / glare-covered / tiny capture with 422 at upload, so the customer retakes it on the spot.
- Debugging: `include_agent_outputs=true`. Advanced: `ocr_grounding=false`, `local_mrz_search=false`, `agent_passes=1|2|3`.

```bash
curl -X POST "http://localhost:8000/universal/extract/sync?priority=live" -F "file=@id.jpg"
curl -X POST "http://localhost:8000/universal/extract/sync" -F "file=@front.jpg" -F "file_back=@back.jpg"
python -m universal front.jpg [back.jpg]                    # no server needed
```

A Postman collection is in [`postman_collection.json`](postman_collection.json) (folder *Universal Engine*: Service, Extract, Jobs, Quality policy checks). If a POST fails with `EISDIR`, allow Postman to read files outside its working directory (Settings → General) and re-select the file.

### What's checked deterministically

| Check | Coverage |
|---|---|
| MRZ | ICAO 9303 TD1 (3×30, ID cards), TD2 (2×36), TD3 (2×44, passports). Found straight from the OCR text of every image, a card's back included. Lines are combined across the two OCR engines by role, and the check digits pick the right combination. Trust is per field: only the number, dates and personal number have check digits; a disputed name line is re-read with per-character voting and never overrides the visual spelling. The slow local MRZ search is skipped on card types that carry no MRZ (employee, student, voter cards). |
| Barcodes | AAMVA PDF417 (US/CA) and the Ethiopian Fayda QR are parsed. Any other barcode's text is surfaced raw. |
| National ID numbers (checksum) | CHN, IND (Aadhaar, Verhoeff), ARE, ZAF, TUR, BRA (CPF), SAU, KOR, ESP, KAZ, SWE (personnummer), BEL (national register number) |
| National ID numbers (structure + embedded date/sex) | EGY, IDN, MYS, PAK, ETH (FAN/FIN), NGA |
| Licence numbers that encode the holder | UK DVLA and Ontario: used to verify the surname and to undo a reversed name split. |
| Document-number formats | NLD, ITA, GBR, ETH, CAN (ON/AB), ARE, SWE, BEL, ESP. Look-alike characters repaired (`CA123450A` → `CA12345OA`); a CAN, a personal number, a phone number or a label word is never taken as the document number. Numbers glued together by the agents are split by the card's own `ID No` label. |
| Calendars, exact | Ethiopian, Persian (Jalali), Thai Buddhist, Minguo, Japanese eras. Dual dates (Ethiopian + Gregorian) are solved as a pair — see [Dates in two calendars](#dates-in-two-calendars). |
| Calendars, approximate | Hijri, tabular (±1–2 days vs. Umm al-Qura), flagged as approximate |
| Calendars, not converted | Hebrew, Nepali Bikram Sambat. Reported with the original kept. |
| Country | The issuing country the card itself names, in any language (`Itiyoophiyaa`, `ኢትዮጵያ`, `مصر`), or whose label languages it uses, overrides an agent's guess. |
| Privacy | Aadhaar numbers are masked to `XXXX XXXX 1234` |

### Reading the result

- **`fields.<name>`** carries `value`, `confidence`, `source`, `verified_by` (every independent check passed), `flags` (why something changed) and `alternatives` (every replaced value).
- **`extracted_data`** is the flat view, plus `given_names`, `full_name_as_printed`, `surname_local` / `given_names_local`, `nationality_code`, and **`additional_fields`**: values the card labels that the schema has no field for (registration / employee number, position, department, branch, sub-city, woreda, phone, emergency contact, occupation…).
- **`languages`**: each language on the card with the printed evidence for it (labels, marker words, script).
- **`verification`**: `printed_text` (names, dates, number decisions), `labels` (every labelled date and value, printed titles, the document-type decision), `country`, `printed_only`, `spelling_by_script`, `mrz`, `barcode`, `ocr_grounding`, `id_numbers`, `dual_calendar`, `plausibility`, `identity_signals`.
- **`human_review_required`** always comes with plain-language **`review_reasons`**, e.g. *"date_of_birth: '15/04/1982' carries no calendar mark while the card's other dates are Ethiopian — 1982-04-15 if Gregorian, 1989-12-24 if Ethiopian"* or *"document_number: agents read '041237', OCR reads '941237' — check the card"*.
- **`overall_confidence`** is the mean confidence of the identity fields; a birth date counts only on documents that carry one (not on employee or student cards). **`completeness_percent`** is the share of schema fields filled.
- **`capture_quality`**, **`queue`** and **`agents`** (per call: passes, knowledge coverage, planned vs. actual tokens, model load / prompt / generate seconds) describe how the result was produced.

### The model reads, the code decides (`universal/printed.py`)

On real documents the agents *read* well but *interpreted* poorly: the surname taken as the last word, 01/02/1984 read day-first on a US card, the CAN taken as the document number, an Ethiopian date passed off as Gregorian. So a short **transcriber** pass (one extra qwen2.5vl call) asks only for label-anchored raw text — the value at each name label, every date exactly as printed (and every date with its label, `dates_with_labels`), the document number as printed, the name in its native script — and code interprets it:

- **Names:** ISO 18013 label numbers decide (field `1` is the surname even when printed below `2`). Honorifics (MR, DR, ATO, W/RO…) and `[a]`-style markers are dropped; "SURNAME, Given" is split; an ALL-CAPS run in a mixed-case name is the surname.
- **Name parts** (`names.py`): the first given word is `first_name`, the rest `middle_name`; the surname is one printed field and never split; particles stay attached (ABDUL RAHMAN, DE LOS ANGELES); romanized CJK given names are never split. With only a "Full Name" printed, order decides only where culture makes it decisive: patronymic (ETH, ERI, SOM: own name · father · grandfather) and family-name-first countries. On a patronymic card the words between the agreed first and last word are the father's name, even when the agents dropped or glued them.
- **Full-name order:** `full_name` reads first + middle + last; the printed order is kept in `full_name_as_printed`. Label words and card titles are never accepted as names.
- **Native-script names:** composed into `full_name_local` in the same natural order, kept only if the issuing country prints that script (`names.LOCAL_SCRIPTS`), and Latin transliterations or mixed-script garbles are rejected.
- **Ge'ez ↔ Latin spelling** (`ethiopic.py`): Ge'ez is a syllabary, so each character romanizes deterministically. Each Amharic word is taken from the reading that spells the Latin name best, and a one-character fix is applied only when exactly one exists (ረና → ሀና for HANA). The check works in both directions: when OCR "corrects" a Latin name (`KEBEDE` → `KEPEDE`), the printed Amharic spelling (ከበደ = ke-be-de) overrules it.
- **Priority:** these interpretations never override the MRZ, a barcode or a checksum-verified value.

Other safeguards, each found on a real document:

| Situation | What the engine does |
|---|---|
| OCR keeps field numbers glued to values (`1DOE`) or a surname ends in a comma (`ROE,`) | Treated as physical label evidence; it outranks the transcriber. |
| The issuing country isn't read, or the agents guess the wrong one | The country the card names, then its label languages, then a distinctive checksum-valid number, then the address format. |
| An agent returns an example value from the prompts | Prompt-echo detection discards it. |
| An agent tacks "EC" onto a Gregorian date on a non-Ethiopian card | A calendar may override a date only where the country prints that calendar. |
| A passport's visual name drops the surname | The MRZ name completes it. |
| OCR glues words (`KEBEDETESFAYE`, `IssueDt`, `DOBPh+07…`) | Spelling correction and label matching both work inside glued tokens. |

`tests/run_id_batch.py <image_dir> <output_dir>` runs the engine over a folder of IDs and writes one JSON per document plus `summary.tsv`. Keep the output directory outside the repository when the images are real.

### Printed labels in every language decide what a value is (`labels.py`, `layout.py`, `languages.py`, `understanding.py`)

- **The label vocabulary** ([`labels.py`](universal/labels.py)): 36 fields in 41 languages across Latin, Ge'ez, Arabic, Cyrillic, Greek, Hebrew, Indic, CJK and Thai scripts, including Amharic, Afaan Oromo, Tigrinya and Somali.
  - Identity: dates of birth / issue / expiry, surname, given names, full name, father / grandfather / mother, sex, nationality, place of birth, blood group, height.
  - Address: region, sub-city / zone, woreda, kebele, house number, address, phone.
  - Work and civic cards: employee / registration number, position, department, branch, grade, occupation, issuing authority, issuer name, signature.
  - **Emergency contact** (name, address, phone): never mixed into the holder's fields.
  - `other_date`: employment, registration, entry dates — never birth, issue or expiry.
  - Matching folds case, accents and the Ge'ez homophone rows (ሠ/ሰ, ፀ/ጸ, ሐ/ኀ/ሀ, ዐ/አ), tolerates one OCR slip (`Expiy Date`, `Cosa Dhiigaa`), works inside glued OCR text, and the label with the most words wins (`Lakkoofsa Bilbilaa` is the phone label, not `Lakkoofsa` + a stray word).
- **The layout** ([`layout.py`](universal/layout.py)) pairs each value with its label by **position on the card**, not OCR text order:
  - same box → same row to the right → directly below / above; labels stacked in three languages with the value beside the middle line; several labels on one address line (`Region: … SubCity: … Woreda: …`), one value each.
  - It copes with tilted photos (overlapping boxes), bilingual labels (`Kutaa Magaalaa / ክፍለ ከተማ` is not a value), and a far box on the same row (a signature block) losing to the value directly below.
  - A name label keeps both its Latin and its native-script reading. Values are validated per field (a blood group, a phone number, digits for numbers).
  - Boxes come from RapidOCR (read in the background while the agents run) plus Tesseract for label scripts RapidOCR can't read (Ge'ez, Arabic…), chosen from the country and scripts the card actually shows.
- **Dates:** a date the agents filed under the wrong field moves to its labelled field; an agents' birth date that the card labels as the issue date is cleared. Only a label in the same box, on the same row or copied by the transcriber may overrule the model, and one printed date never fills two fields.
- **Document type:** the card's own title (`Employee ID Card`, `የሰራተኛ መታወቂያ`, `Waraqaa Eenyummaa Jiraattota`), its statement ("The bearer of this ID card is an employee of our Bank") and its work-card fields decide. A generic title never overrides a specific agent type that the card supports, but it does override one that nothing printed supports. A birth date is required only on state identity documents.
- **Languages** ([`languages.py`](universal/languages.py)): every agent votes, the fast scanner included; the printed evidence (label languages, marker words like `Oromiyaa`, scripts by country) decides. `language_primary` / `language_secondary` are filled and `languages` lists each with its evidence.
- **Organisation:** a work or student card's issuer is read from its header (the website line is ignored).

### Printed only: nothing that isn't on the card

Local vision models fill gaps with plausible inventions: on real cards they returned an Egyptian issuing country for an Oromia card, "Djiboutian" for a card printing `Nationality: Ethiopian`, the card's title as the holder's name, and a phone number as the personal number. Two layers stop this.

**In the prompts** (rule 11, *PRINTED ONLY*, sent to all five agents and the supervisor): report a value only if it is printed on this document; never infer from the country, city, name, face, language or document type; nationality only from a printed nationality field or the MRZ; a Tel / phone number is never a personal number; never use placeholder text ("woman's name", "N/A").

**In code** (`understanding.require_printed`, `decide_country`, `drop_titles_as_names`), because prompts alone are not enough:

| Field | Rule |
|---|---|
| `nationality` | The value beside the card's nationality label wins. Otherwise it must be printed (country name, code, demonym or native name, in any script) or MRZ-verified, else it is dropped with flag `not_printed`. |
| `issuing_country` | The country the card names, or whose label languages it uses, overrides an agent's guess that the card never names. |
| `full_name`, `full_name_latin` | The name beside the name label wins. A card title — even misspelled — is never a name; if no name can be read, it goes to review. An unprinted "Latin name" is replaced by the printed one. |
| `personal_number` | A number printed beside a phone label (one misread digit tolerated) moves to `additional_fields.phone`. |
| `blood_type`, `region`, `district`, `sub_district` | The labelled value wins; `+` alone is not a blood group. |
| `date_of_birth_orig` | An agent's "original-calendar date" is used only if that date is printed. |
| `document_number` | When OCR reads a near-identical number, it is shown to the reviewer — never swapped in. |

### Dates in two calendars

Ethiopian cards print many dates twice, Ethiopian and Gregorian, often Gregorian month-first, often damaged by OCR. The rules, in order, decide only when the evidence is unambiguous:

1. **Calendar mark printed** (`25/10/2016 E.C`, `ዓ.ም`, `AH`): converted by arithmetic; the mark is kept across OCR engines even when one loses it. Several marked dates transcribed into one field are separated and placed.
2. **Month names in two calendars** (`የካቲት 07, 1987 / FEB 14, 1995`, glued forms like `Mar 192026`): each half is read; they must agree by arithmetic. If the Ge'ez month was lost, it is accepted only when exactly one month makes the halves agree.
3. **Calendar-pair solver** for damaged numeric pairs (`15/41982` beside `12/241989` → 15/04/1982 EC = 1989-12-24): a Gregorian date is accepted only when an Ethiopian reading on the same row converts to it **and** a Gregorian reading supports it, and exactly one date satisfies both.
4. **No mark, on a card whose other dates are marked EC:** neither reading is chosen; both go to review.
5. **Contradictions:** halves that disagree go to review; when issue and expiry contradict, the unprinted one is dropped, never the printed one.

### Token budgets and multi-pass agents (`universal/multipass.py`)

Each agent call is packed to the model's real token budget:

```
text budget = context limit − image tokens − output reserve − safety margin
```

All four numbers are measured against the model itself (`universal/tokens.py`):

- **Context limit:** read from Ollama's model info. Large-context models are capped at 8K, because 16K made qwen2.5vl 7.8 GB and forced Ollama to evict and reload models between agents on this Mac.
- **Image tokens:** probed once per model. They're fixed per model: llava 577, moondream 729.
- **Text tokens:** an estimate calibrated per model on a sample that fits the model's own window (a longer sample is truncated by Ollama and measures too few tokens — that once halved moondream's estimates and made the fast scanner fail). A ratio outside 1–6 bytes/token is rejected. Calls more than 85 % full are counted exactly through Ollama.
- **Self-healing:** if Ollama still refuses a call as too large, its error states the true size (`n_prompt_tokens`). The engine scales that model's estimates up, re-plans and retries once (`replanned_after_overflow` in the agent log), so a wrong estimate costs one retry, never the agent.

**Packing.** The prompts are split into blocks at their natural seams: `═══` sections, paragraphs, and individual list items. Blocks are then chosen as follows:

- The identity, extraction rules, output format, placeholders and return instructions always go in.
- The remaining blocks are scored for this document: the fields the call asks for, the detected script and country, the document type, and topics such as MRZ, address, calendars and names.
- Blocks fill the call in score order until the budget is reached. On an Ethiopian ID, the Ethiopic, Amharic and Kebele/Woreda rules win the space; the Cyrillic and CJK rules are dropped first.

**Passes** (`agent_passes=auto|1|2|3`, the API query parameter or `--passes` on the CLI). With more than one pass, the schema is split into field groups (identity, document, location, machine-readable, metadata). The same model runs once per group, each pass carrying the knowledge relevant to its own fields, and the results are merged. `auto` behaves like this:

- If everything relevant fits in one call, it makes one call.
- It splits only when splitting gives every pass at least 80 % of its relevant knowledge.

This rule comes from a measurement: moondream run 3 times at about 30 % coverage per pass read *fewer* fields than one maximally packed call. With the current prompts, `auto` therefore makes one fully packed call for every agent. llava now gets its complete relevant prompt, about 3,050 of 4,096 tokens, where it used to get about 1,960. Forcing `2` or `3` is still available for experiments or for longer custom prompts.

**Template check.** moondream's chat template (`Question: {{ .Prompt }} Answer:`) has no system slot, so Ollama silently dropped every system prompt sent to it. Models whose template lacks `.System` now get the system text folded into the user message. The prompts file's own `build_ollama_message()` does the same for the families listed in `MODELS_WITHOUT_SYSTEM_SLOT`. This is detected from the template automatically, so it also covers future models.

The per-agent log in the result shows `passes`, `knowledge_coverage`, and planned vs. actual prompt tokens for each pass. `timings_seconds` breaks the run down by stage.

### Serving at high traffic

**Quality first, speed second.** Speed comes from the architecture around the agents, never from fewer of them. Every document gets:
- the full agent team (5 agents plus the transcriber)
- the same models and prompts
- full-resolution images
- the whole verification layer

What changed is everything *between* the model calls. Diagrams are in [docs/UNIVERSAL_ARCHITECTURE.md §8](docs/UNIVERSAL_ARCHITECTURE.md#8-serving-at-high-traffic).

| Layer | Module | What it does | Quality impact |
|---|---|---|---|
| Durable priority queue | [`jobqueue.py`](universal/jobqueue.py) | SQLite (one machine) or PostgreSQL (`FOR UPDATE SKIP LOCKED`, many nodes). Jobs survive restarts; a crashed worker's job is retried when its lease expires; images travel inside the queue and are deleted when the job finishes. | none |
| Priority tiers | `jobqueue.py` | `live` > `online` > `batch` > `bulk`. Lower tiers age forward (10 min / 1 h / 6 h), so they are never starved. | none |
| One call per GPU | [`inference.py`](universal/inference.py) | An inference host runs **one** model call at a time. Concurrent calls on one GPU produced degenerate output on this project's hardware, so the engine never does it. | protects quality |
| Hot models | `inference.py` | `keep_alive=-1` on **every** request (including the token-count probes, which used to reset it to Ollama's 5-minute default), plus a warm-up when workers start. | none |
| Scale out | `inference.py`, [`worker.py`](universal/worker.py) | `UNIVERSAL_OLLAMA_HOSTS`: more GPUs, each running the whole team. `UNIVERSAL_MODEL_HOSTS`: pin a model to its own GPU, so documents flow through the models like an assembly line. `python -m universal.worker` adds worker processes on any machine. | none |
| Cheaper planning | [`multipass.py`](universal/multipass.py) | When the exact token count says a packed prompt is slightly over budget, the lowest-value blocks go. The number to drop is now found with a guided search (usually 3 exact counts) instead of one re-count per dropped block. On llava:13b each count is a ~5 s prompt evaluation, and the verifier used to make 5 in a row. Candidate plans that already fail the 80 % coverage rule on their estimate are also rejected without counting. **Tested byte-identical** to the previous prompts. | identical prompts |
| Double submissions | `service.py` | A **byte-identical** upload with identical options, sent while the first is still queued or running, joins that job. Similar-looking documents never do. | none |
| Capture check | [`capture.py`](universal/capture.py) | Blur, exposure, contrast, glare and resolution are measured in milliseconds. The grade goes into the result; `reject_poor_capture=true` refuses a poor capture at upload. The images the agents see are **not** altered: no denoising, and no super-resolution, which can invent strokes. | the customer retakes a bad photo |
| Identity signals | [`signals.py`](universal/signals.py) | The same document number seen with a different identity, or the same photo (re-saved or edited) reused for a different identity, sends the result to human review. The **byte-identical file** sent again is reported as information only (`same_file_submitted_before`), since a different reading of it is the engine's, not a second person. Stored only as HMAC digests, SHA-256 and perceptual hashes; on PostgreSQL, pgvector's HNSW Hamming search. **Flags only — never fills a field.** | adds fraud review |
| Observability | [`metrics.py`](universal/metrics.py) | `/universal/metrics` (Prometheus): queue wait, stage and agent times, model **load** time per call (swapping), review rate, capture grades, signals. `/universal/health`: models resident and fully on the GPU. | measures it |
| Quality gate | [`tests/quality_gate.py`](tests/quality_gate.py) | Compares a run before and after a change, field by field. A change that loses a field is rejected. | enforces it |

#### Database: PostgreSQL + pgvector, local in one command

One database holds everything the serving layer keeps:
- the job queue and results
- the identity-signal store
- its pgvector index

It's the same engine on this Mac as on a bank server, so production is only a connection-string change.

```bash
python -m universal.localdb setup        # once: installs PostgreSQL + pgvector (conda-forge), initialises, secures, starts
python -m universal.localdb use "postgresql://user:pass@db-host:5432/kyc?sslmode=require"   # OR: a PostgreSQL you already run
python -m universal.localdb status       # server / pgvector version, table sizes
python -m universal.localdb autostart on # start with the Mac at login (launchd)
python -m universal.localdb backup       # compressed pg_dump into var/universal/backups/ (personal data: store encrypted)
python -m universal.localdb psql | url | start | stop
```

- **No Docker, Homebrew or root.** The binaries live in their own conda env (`kyc-pgvector`), separate from `myenv`.
- **Data** is in `var/universal/pgdata` (0700, git-ignored).
- **Network:** listens on 127.0.0.1:5433 only; no Unix socket.
- **Auth:** scram-sha-256 with a random password, stored only in `var/universal/database.env` (0600).
- **After `setup` nothing else needs configuring.** The engine reads `database.env` and starts the server if it's down. `/universal/health` shows the backend in `database`.
- Without PostgreSQL, the engine falls back to SQLite files and says so at startup.
- **Already running PostgreSQL** (e.g. the bank's server)? Use `use` instead of `setup`. It checks the connection and confirms the pgvector extension is installed on that server. If it isn't, it says exactly what the administrator must install. It then enables pgvector, creates the engine's tables, warns when a remote connection isn't encrypted, and records the URL in `database.env`.

**Deliberately not done.** Result caches keyed on the document number or on image similarity: two people's cards of the same type look alike, so a similarity hit can return someone else's identity. The only cache is an opt-in exact-byte one (`UNIVERSAL_EXACT_CACHE_SECONDS`, off by default). It reuses only review-free results, and every engine or prompt change invalidates it.

**Configuration** (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `UNIVERSAL_DATABASE_URL` / `UNIVERSAL_QUEUE_URL` | the local PostgreSQL from `localdb setup` (SQLite if none) | `postgresql://…` of a shared server for multi-node |
| `UNIVERSAL_OLLAMA_HOSTS` | `OLLAMA_HOST` or local | comma-separated inference hosts |
| `UNIVERSAL_MODEL_HOSTS` | — | `llava:13b=http://gpu3:11434;qwen2.5:7b=http://gpu4:11434` |
| `UNIVERSAL_HOST_SLOTS` | `1` | calls per host at once — keep 1 unless a host passes the quality gate at more |
| `UNIVERSAL_WORKERS` / `UNIVERSAL_INPROCESS_WORKERS` | one per GPU host / `auto` | documents in flight; `0` in-process makes an API-only node |
| `UNIVERSAL_PREVENT_SLEEP` | `1` | a worker keeps the Mac awake (caffeinate) while it runs |
| `UNIVERSAL_KEEP_ALIVE` | `-1` | never unload a team model |
| `UNIVERSAL_ALLOW_FAST_MODE` | `0` | quality policy: full team only |
| `UNIVERSAL_RESULT_TTL` | `86400` | seconds a finished result is kept (personal data) |
| `UNIVERSAL_IDENTITY_SIGNALS` / `UNIVERSAL_SIGNALS_KEY` / `UNIVERSAL_SIGNALS_URL` | on / generated key file / queue DB | identity-consistency checks |
| `UNIVERSAL_EXACT_CACHE_SECONDS` | `0` (off) | exact-byte result reuse |
| `UNIVERSAL_SEED` | unset | fixed sampling seed: the same document gives the same result on every run. Measured: two full-team runs of one ID with a seed gave byte-identical output from all six agents; without it, unchanged code scored 311 and 320 of 329 fields on two runs. Recommended for KYC auditability and an exact quality gate. |
| `UNIVERSAL_PRIORITY_AGING` | `0,600,3600,21600` | seconds each tier yields to newer higher-tier work |

**One document per GPU** is the measured default. On this Mac, two workers gave the same throughput (108 s per document) but made each document take about twice as long (173–261 s), because the model calls are the bottleneck and interleaving two documents makes the models swap more. More documents in flight come from more GPUs (`UNIVERSAL_OLLAMA_HOSTS`) or from pinning models to their own GPUs (`UNIVERSAL_MODEL_HOSTS`).

Workers also keep the Mac awake while they run. An idle-sleeping laptop once stalled a document for 3 hours, and a lease that ran out during the sleep let a second worker take over a job that was still running. Workers now renew their own leases first.

### The prompts file

`universal_id_extraction_prompts.py` defines each agent's prompt text and model, and `universal/prompts.py` reads it at load time. Keep these points in mind when editing it:

- **Languages by script.** The system prompt lists every language under the script it is written in (Latin, Cyrillic, Arabic, Ge'ez, Devanagari, other Indic, Southeast Asian, CJK…). Afaan Oromo is under Latin (Qubee), with its main labels. Each script is its own block, so the packer keeps the scripts the document actually shows.
- **Rule 11, PRINTED ONLY**, in both system prompts, plus the supervisor's *never guess* rule: see [Printed only](#printed-only-nothing-that-isnt-on-the-card). Keep it — it is the prompt half of the anti-invention design.
- **Two system prompts.** `SYSTEM_PROMPT` (~7.9 KB) and `SYSTEM_PROMPT_COMPACT` (~1.2 KB) for moondream (2,048-token context) and llava (4,096). After any edit, check that every agent still plans one pass at full coverage (the agent log shows `passes` and `knowledge_coverage`).
- **Ethiopian address mapping:** printed Region → `region`, sub-city / zone → `district`, woreda (+ kebele) → `sub_district`.
- **Placeholders.** `build_ollama_message()` fills `{previous_extraction}`, `{verified_extraction}` and `{agent_outputs}` by plain replacement, not `str.format()`, so the JSON braces in the prompts are safe.
- **TD3 check digits.** The verifier prompt uses the ICAO positions: characters 10, 20, 28 and 44 of line 2.

### Tests

```bash
python -m pytest tests/ -q            # offline tests, no Ollama needed
python tests/make_specimens.py        # regenerate the synthetic fixtures in tests/fixtures/
UNIVERSAL_TEST_PG_URL=postgresql://user@host/db python -m pytest tests/test_universal_serving.py -q   # + PostgreSQL/pgvector

# Quality gate — run BEFORE and AFTER any performance change, outside the repo (results hold personal data)
python tests/run_id_batch.py ids/ ~/kyc-runs/before
python tests/run_id_batch.py ids/ ~/kyc-runs/after
python tests/quality_gate.py --baseline ~/kyc-runs/before --candidate ~/kyc-runs/after --truth ~/kyc-runs/truth.json
```

The gate prints accuracy for both runs, every field lost or gained, every changed value, the review rate and the timings. It exits non-zero if quality dropped.

The fixtures are fictional: the ICAO "Utopia" passport and TD1/TD2 specimens, a bilingual Amharic/English card with dual calendars, a Chinese resident ID carrying the widely published sample number, a trilingual (Amharic / Afaan Oromo / English) employee card, and a bank staff card front and back with stacked labels and `E.C` dates.

**Two kinds of tests, both needed:**

- **Rule tests** (`pytest`, ~440 offline): every rule above has a test built from the OCR text of a real card that exposed it, with fictional names.
- **A representative test set** (`run_id_batch.py` + `quality_gate.py`): the only way to measure "universal". The current 31-ID set is mostly US / EU / Ethiopian licences and passports; it does not represent Ethiopian kebele and regional resident IDs, Fayda, employee cards or foreign residence cards. Build a set of 5–10 real samples per document type you serve, with the correct answers in a truth file, **outside the repository**, and run it with `UNIVERSAL_SEED` set before every change is accepted. Without a seed, unchanged code varies by several fields between runs.

### Current limits (measured, not hidden)

| Limit | Effect | What helps |
|---|---|---|
| Ge'ez on small, glossy or low-resolution cards | `full_name_local` often misspelled or left empty | a sharper capture; a Ge'ez-capable OCR model |
| Coloured ink (e.g. a red `A+` blood group), security patterns | the value can't be OCR-confirmed and may be dropped | capture guidance; the field is left empty, never invented |
| One digit misread by the agents (`041237` vs printed `941237`) | sent to review with both readings | a person decides; a larger vision model |
| Documents never seen in testing | new layouts can expose new gaps | the representative test set above |
| Speed on a 24 GB Mac | ~90–150 s per document (models swap) | a GPU server, or a dedicated host for llava (`UNIVERSAL_MODEL_HOSTS`) |



---

## Usage Examples

```bash
# Auto-detect document type, wait for the full result (production: use callback_url instead)
curl -X POST "http://localhost:8000/extract/id/sync?doc_type=auto" \
  -F "file=@ethiopian_id.jpg"

# Known document type — skips the wasted passport-MRZ search, much faster
curl -X POST "http://localhost:8000/extract/id/sync?doc_type=national_id" \
  -F "file=@ethiopian_id_front.jpg" -F "file_back=@ethiopian_id_back.jpg"

# Production pattern: fire-and-forget with a webhook callback
curl -X POST "http://localhost:8000/extract/id?doc_type=drivers_license&callback_url=https://your-service/callback" \
  -F "file=@dl_front.jpg" -F "file_back=@dl_back.jpg"
```

```bash
# Python test client
python test_client.py passport.jpg passport sync
python test_client.py ethiopian_id.jpg national_id sync
python test_client.py dl_front.jpg drivers_license sync dl_back.jpg
```

---

## Response Format

`extracted_data` is a flat convenience view; `document` is the full
provenance-carrying shape — every field is `{value, raw, state,
confidence, source, source_tier, agreeing_sources}`, not a bare string.
`state` is one of `read` / `unreadable` / `printed_blank` /
`absent_from_specimen` / `not_applicable` — check `state == "read"`
before trusting `value`.

```json
{
  "job_id": "abc-123",
  "status": "completed",
  "pipeline": "idx_pipeline",
  "document_type": "national_id",
  "extraction_method": "barcode+ocr",
  "confidence": "high",
  "coverage_tier": "A",
  "human_review_required": false,
  "warnings": [],
  "extracted_data": {
    "full_name": "MESAY KEBBEDE LEMMA",
    "id_number": "5304732902634061",
    "date_of_birth": "1996-11-28",
    "gender": "M"
  },
  "document": {
    "document_category": "national_id",
    "document_number": {
      "value": "5304732902634061", "raw": "5304732902634061",
      "state": "read", "confidence": 0.9,
      "source": "barcode", "source_tier": 1, "agreeing_sources": []
    },
    "full_name": {
      "value": "Mesay Kebbede Lemma", "raw": "Mesay Kebbede Lemma",
      "state": "read", "confidence": 0.55,
      "source": "label_anchor", "source_tier": 3, "agreeing_sources": []
    },
    "verification_checks": {"dob_dual_calendar_plausible": true},
    "mrz_checks": {}, "warnings": [], "human_review_required": false
  },
  "validation": {"suspicious": false, "reasons": []},
  "total_duration_seconds": 12.3
}
```

---

## Known issues on this machine

- **GPU memory is the bottleneck, not compute**: the 24 GB of unified
  memory is shared by the CPU and GPU, and Ollama's Metal backend can use
  about 17.8 GB of it. The four models (qwen2.5vl:7b, llava:13b,
  qwen2.5:7b, moondream) don't all fit at once, so Ollama evicts and
  reloads them between agents. A reload also happens whenever a model is
  called with a different context size. The universal engine therefore
  caps large models at an 8K context (at 16K, qwen2.5vl grew to 7.8 GB)
  and keeps its token-probe calls on the same context size as the real
  calls. A full universal run takes about 90–150 s per document (fast mode is disabled by policy).
  Avoid running other memory-heavy apps during extraction.
- **Laptop sleep stalls long runs**: on battery, macOS idle-sleep froze
  one request for 12 minutes mid-batch. `tests/run_id_batch.py` holds a
  `caffeinate -i` assertion for its own lifetime. For the API server, start
  it with `caffeinate -i python main.py`.
- **Occasional "token repeat limit reached"**: this is Ollama's abort on
  degenerate repeating output. The universal engine's `chat_json()`
  resamples on the GPU with a stronger repeat penalty. Retrying on the CPU,
  which suited the old Intel Vulkan machine, left the model CPU-loaded on
  Apple Silicon, and every later call ran about 10 times slower. It now
  falls back to the CPU only on real device-loss errors, with
  `keep_alive=0` so the CPU runner unloads immediately. The existing
  pipeline's `ollama_client.vision_chat()` (used by `crewai_pipeline.py` and
  `idx/vision_ollama.py`) follows the same rule.
- **`auto` doc_type pays for a wasted MRZ search** on non-passport
  documents before reaching the right tier (e.g. ~40s TD3 search on an
  Ethiopian Digital ID, which has no MRZ, before the fast barcode tier
  runs). If you know the document type, pass it explicitly — this is
  the single biggest speed win available today.
- Local vision-model accuracy (channels 3-4) on dense, security-patterned
  documents is genuinely limited on this hardware — `validators.py`'s
  `sanity_check()` flags implausible results (institutional-keyword
  names, malformed MRZ, cross-field contradictions) but can't fix the
  underlying read. Barcode/MRZ-sourced fields (channels 1-2) don't have
  this problem — they're checksum-verified, not model guesses.

Machine: MacBook Pro (Mac17,9) | Apple M5 Pro, 15 cores | 24 GB unified memory | Ollama on Metal
# ID-exract-with-AI-agent
# kyc-extrcat-with-AI-agents
