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

A second, additive engine in [`universal/`](universal/). The `/extract/*`
pipeline above is untouched; the only change to existing code is two lines in
`main.py` that mount the router. The prompts come from
[`universal_id_extraction_prompts.py`](universal_id_extraction_prompts.py),
which stays the single source of truth for prompt text and model assignment.

📐 **Diagrams:** [docs/UNIVERSAL_ARCHITECTURE.md](docs/UNIVERSAL_ARCHITECTURE.md) shows how the agents
communicate (including the transcriber), how every value is verified, and how the token limit of the
small models is solved.

The design idea: **the agents propose values, and deterministic code checks
them.** Local vision models misread characters, confuse calendars and
sometimes invent plausible values. So everything they say is checked against
evidence that doesn't come from a language model.

```
barcode decode (zxing)  ─┐   verified barcode data is fed to the extractor as ground truth
                         ▼
moondream → qwen2.5vl → llava:13b → qwen2.5vl → qwen2.5 supervisor      ← RapidOCR runs in
 scanner    extractor    verifier    layout      (text-only)               parallel on CPU
                         ▼
MRZ check digits (agents' lines → else local OCR re-read)
national-ID checksums + birth date / sex encoded in the number
OCR grounding (RapidOCR + tesseract in the document's script):
    hallucinations dropped · dual-calendar dates fixed · truncated numbers completed
    · OCR picks between the agents' own candidates · spelling near-misses corrected
calendar arithmetic · ISO 3166 countries (incl. native names) · plausibility · expiry
                         ▼
per field: value · confidence · source · agreeing_agents · verified_by · flags · alternatives
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/universal/info` | Agents + model status, checksum schemes, calendars, OCR engines |
| POST | `/universal/extract?mode=full` | Async (`wait=true` to block, `callback_url=` webhook) |
| POST | `/universal/extract/sync?mode=fast` | Sync |
| GET | `/universal/status/{job_id}` | Poll |
| DELETE | `/universal/job/{job_id}` | Remove job |

- `mode=full`: the 5-agent team, about 1.5–2 min on this machine.
- `mode=fast`: a single qwen2.5vl read, about 15 s.
- Both modes run the whole verification layer.
- `doc_type` is only a hint to the agents. It accepts `passport`, `national_id`, `drivers_license`, `residence_permit`, `visa`, `refugee_travel_document`, `military_id` and `other`.
- Optional flags: `ocr_grounding=false`, `local_mrz_search=false`, and `include_agent_outputs=true` (for debugging).

```bash
curl -X POST "http://localhost:8000/universal/extract/sync?mode=fast" -F "file=@id.jpg"
curl -X POST "http://localhost:8000/universal/extract/sync?mode=full" -F "file=@front.jpg" -F "file_back=@back.jpg"
python -m universal id.jpg [back.jpg] --mode fast          # no server needed
```

### What's checked deterministically

| Check | Coverage |
|---|---|
| MRZ | ICAO 9303 TD1 (3×30, ID cards), TD2 (2×36), TD3 (2×44, passports). Found straight from the OCR text of every image, a card's back included, without depending on the agents to transcribe it. Lines are combined across the two OCR engines by role, and the check digits pick the right combination. Trust is per field: only the number, dates and personal number have check digits. The name line has none, so when the engines disagree on it, the local MRZ search re-reads it with per-character voting. A disputed MRZ name never overrides the visual spelling, but a cleanly read one settles small visual misspellings ("LEMA" becomes "LEMMA"). The MRZ also decides which tokens are the surname. |
| Barcodes | AAMVA PDF417 (US/CA) and the Ethiopian Fayda QR are parsed. Any other barcode's text is surfaced raw. |
| National ID numbers (checksum) | CHN, IND (Aadhaar, Verhoeff), ARE, ZAF, TUR, BRA (CPF), SAU, KOR, ESP, KAZ, SWE (personnummer), BEL (national register number) |
| National ID numbers (structure + embedded date/sex) | EGY, IDN, MYS, PAK, ETH (FAN/FIN), NGA |
| Licence numbers that encode the holder | UK DVLA (surname letters, birth date, sex, initials) and Ontario (surname initial, birth date, sex). Used to verify the surname, and to swap surname and given names back when an agent reversed them. |
| Document-number formats | NLD, ITA, GBR, ETH, CAN (ON/AB), ARE, SWE, BEL, ESP. Look-alike characters are repaired (`CA123450A` becomes `CA12345OA`), and a 6-digit CAN, a personal number or a label word is never taken as the document number. |
| Calendars, exact | Ethiopian, Persian (Jalali), Thai Buddhist, Minguo, Japanese eras |
| Calendars, approximate | Hijri, tabular (±1–2 days vs. Umm al-Qura), flagged as approximate |
| Calendars, not converted | Hebrew, Nepali Bikram Sambat. Reported with the original kept. |
| Privacy | Aadhaar numbers are masked to `XXXX XXXX 1234` |

### Reading the result

- **`fields.<name>.confidence`** is based on evidence. Agreement between agents tops out at 0.85. Checksum, barcode or OCR corroboration raises it, and each source is listed in `verified_by`.
- **`overall_confidence`** is the mean confidence of the identity fields.
- **`completeness_percent`** is the supervisor prompt's Rule 8: the share of schema fields that were filled.
- **`human_review_required`** comes with explicit `review_reasons`. Examples: a verified source disagreeing with the agents, a checksum failure, an implausible date, or a critical field OCR couldn't corroborate.
- **`alternatives`** always keeps the value that was replaced, so no correction is silent.

### The model reads, the code decides (`universal/printed.py`)

On a batch of 31 real documents (US and Canadian licences, EU ID cards and residence permits, UK, Australian and Ethiopian licences, passports), the agents *read* well but *interpreted* poorly. Typical mistakes:

- taking the surname to be the last token of the name
- reading 01/02/1984 day-first on a US card
- taking the CAN or a personal number as the document number

So a short **transcriber** pass (one extra qwen2.5vl call) asks only for label-anchored raw text: the value at the surname label (`1`, `LN`, `Surname`, `Cognome`...), the value at the given-names label (`2`, `FN`, `Voornamen`, `Nome`...), each date exactly as printed, and the document number as printed. Code then interprets these:

- **Names:** labels follow ISO 18013 numbering (field `1` is the surname even when it's printed below field `2`). Honorifics (MR, DR, ATO, W/RO...) and `[a]`-style markers are dropped. "SURNAME, Given" comma formats are split. An ALL-CAPS run in a mixed-case name ("DOE Jane Mary") is the surname.
- **Name parts** (`universal/names.py`): each word of the given names is its own name. The first word is `first_name`, the rest are `middle_name`, and `given_names` holds them all. The surname is one printed field and is never split ("GEBRE MEDHIN", "GARCÍA LÓPEZ", "VAN DER BERG"). Name particles stay attached to their name (ABDUL RAHMAN, ABD AL RAHMAN, NOUR EL DIN, ABU BAKR, DE LOS ANGELES), a hyphenated name is one name, and romanized Chinese/Korean/Japanese given names are never split into a middle name. When no surname is labelled, name order alone decides only where the culture makes it decisive: patronymic names (ETH, ERI, SOM: given, father's, grandfather's) and family-name-first countries.
- **Full-name order:** once the parts are settled, `full_name` always reads first + middle + last ("MESAY KEBBEDE LEMMA"), whatever order the card prints; US and Canadian cards print the surname first. Family-name-first cultures keep family-first. The order printed on the card is kept in `full_name_as_printed`, with label junk such as "NAME" or an MRZ-line copy cleaned out. Label words returned in place of a name (NAME, SURNAME, NOM, NAAM…) are never accepted as names.
- **Native-script names:** the transcriber also reads the surname, the given names and the whole name as printed in the document's own script (Amharic/Ge'ez, Arabic, Chinese, Cyrillic…). They're composed into `full_name_local` in the same natural order as the Latin name, plus `surname_local` and `given_names_local`. Chinese, Japanese and Korean names are joined without a space (王 + 芳 = 王芳). A Latin transliteration in these fields, or a mixed-script garble ("pora ተጋባ ሕክታራ"), is rejected. A native-script name is also kept only if the issuing country actually prints that script (`names.LOCAL_SCRIPTS`: Ethiopia Ethiopic, the Gulf states Arabic, China/Japan/Korea CJK, Russia Cyrillic, India its Indic scripts…). Models otherwise invent a transliteration: on a 31-document batch they rendered Latin names on US, Canadian and EU cards into Amharic, Cyrillic and Bengali.
- **Ge'ez checked against the Latin spelling** (`universal/ethiopic.py`): Ge'ez is a syllabary, so each character romanizes deterministically (ሀ = ha, ረ = re, ከ = ke, ካ = ka), with gemination allowed (ከበደ = KEBBEDE). Every Amharic word is checked against the Latin name on the same document:
  - Among all readings (transcriber, extractor, OCR), each word is taken from the reading that spells it best.
  - A single wrong character is corrected when exactly one fix makes the word spell the Latin name (ረና becomes ሀና for HANA; ካበደ becomes ከበደ for KEBEDE).
  - Truly ambiguous characters are left alone: የ and ዬ both spell "ye".
- **Naming culture:** the holder's nationality decides the name order when the document gives no structure (an Ethiopian national's UAE card gives given / father's / grandfather's name). A foreign document's surname label is never overruled.
- **Dates:** each country's convention applies (US month-first). A printed dual-calendar pair is resolved by arithmetic, which also settles day/month order: on Ethiopian licences the Gregorian half is printed month-first.
- **Priority:** these interpretations never override the MRZ, a barcode, or a checksum-verified value. On passports the MRZ decides which tokens are the surname.

Transcribing inside the long extraction prompt didn't work: the model just echoed its positional guess. That's why the transcriber is a separate call.

Other safeguards found by diagnosing each real document individually:

| Situation on a real document | What the engine does |
|---|---|
| OCR keeps field numbers glued to values (`1DOE`, `2JANE QUINN`) or a surname line ends in a comma (`ROE,`) | Treats this as physical label evidence. It outranks the transcriber's reading, even when the agents misread the name itself. |
| The issuing country isn't given (the logo is cursive, or the extractor failed) | Uses evidence in this order: the agents, then a distinctive checksum-valid number (`784-…` proves ARE), then the printed header, then the address format (a Canadian postal code, or a US state + ZIP). It never uses a "Nationality:" line. |
| An agent tacks "EC" onto a Gregorian date on a non-Ethiopian card | A calendar may only override a date when the issuing country actually prints that calendar. A mislabelled copy is dropped. |
| An agent returns an example value from the prompts (`15/Meskerem/1982 EC`, `MESAY TADESSE`) | Prompt-echo detection discards it (names, dates and numbers only). |
| Two different dates transcribed into one field | The extra date is placed by plausibility: after birth and not in the future means issue; in the future means expiry. |
| An ambiguous date whose one reading is in the future | A birth or issue date can't be in the future, so the other reading is used. |
| OCR glues words (`KEBEDETESFAYE`) | Spelling correction also searches inside glued OCR tokens. |
| A passport's visual name drops the surname | The MRZ name, which holds every token plus the surname, completes it. |
| A licence number encodes the surname initial (Ontario, DVLA) but no name token fits the agents' split | The structure is rebuilt around the token that carries the encoded letters. |

`tests/run_id_batch.py <image_dir> <output_dir>` runs the engine over a folder of IDs and writes one JSON per document plus `summary.tsv`. Keep the output directory outside the repository when the images are real.

### Token budgets and multi-pass agents (`universal/multipass.py`)

Each agent call is packed to the model's real token budget:

```
text budget = context limit − image tokens − output reserve − safety margin
```

All four numbers are measured against the model itself (`universal/tokens.py`):

- **Context limit:** read from Ollama's model info. Large-context models are capped at 8K, because 16K made qwen2.5vl 7.8 GB and forced Ollama to evict and reload models between agents on this Mac.
- **Image tokens:** probed once per model. They're fixed per model: llava 577, moondream 729.
- **Text tokens:** an estimate calibrated per model, checked exactly through Ollama whenever a call is more than 85 % full.

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

### The prompts file

`universal_id_extraction_prompts.py` defines each agent's prompt text and model, and `universal/prompts.py` reads it at load time. Keep these points in mind when editing it:

- **Two system prompts.** `SYSTEM_PROMPT` is about 1,800 tokens. `SYSTEM_PROMPT_COMPACT` is about 300 tokens and goes to moondream (2,048-token context) and llava (4,096), which can't fit the full one.
- **Placeholders.** `build_ollama_message()` fills `{previous_extraction}`, `{verified_extraction}` and `{agent_outputs}` by plain replacement, not `str.format()`, so the JSON braces in the prompts are safe.
- **TD3 check digits.** The verifier prompt uses the ICAO positions: characters 10, 20, 28 and 44 of line 2.

### Tests

```bash
python -m pytest tests/ -q            # offline tests, no Ollama needed
python tests/make_specimens.py        # regenerate the synthetic fixtures in tests/fixtures/
```

The fixtures are fictional: the ICAO "Utopia" passport specimen, a bilingual
Amharic/English card with dual calendars, and a Chinese resident ID carrying
the widely published sample number.

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
  calls. A full 5-agent run takes about 90–120 s; fast mode takes about 15 s.
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
