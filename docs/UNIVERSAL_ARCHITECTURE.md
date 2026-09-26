# Universal Engine — How the Agents Work Together

This document explains the universal ID extraction engine (`universal/`, served
at `/universal/*`) in diagrams: how the AI agents pass work to each other, where
the **transcriber** fits, how every value is checked by deterministic code, and
how the engine gets full use out of small-context models.

> One idea runs through the whole design: **the models read, the code decides.**
> Vision models are good at reading what is printed and unreliable at
> interpreting it (name order, date conventions, calendars, which number is the
> document number). So the models transcribe and propose, and deterministic code
> interprets and verifies against evidence that doesn't come from a language
> model: check digits, barcodes, OCR, calendar arithmetic.

Contents:

1. [End-to-end flow](#1-end-to-end-flow)
2. [How the agents talk to each other](#2-how-the-agents-talk-to-each-other)
3. [The transcriber: the model reads, the code decides](#3-the-transcriber-the-model-reads-the-code-decides)
4. [Solving the token limit of small models](#4-solving-the-token-limit-of-small-models)
5. [Evidence precedence: who wins a disagreement](#5-evidence-precedence-who-wins-a-disagreement)
6. [From raw readings to name parts](#6-from-raw-readings-to-name-parts)
7. [Reading the result](#7-reading-the-result)

---

## 1. End-to-end flow

```mermaid
flowchart TD
    IMG[/"📄 Front image (+ optional back)"/] --> PRE

    subgraph PRE["① Deterministic pre-pass (≈0.5 s, CPU)"]
        BC["Barcode / QR decode (zxing)<br/>AAMVA PDF417 · Fayda QR"]
    end

    IMG --> OCR
    subgraph OCR["② Early OCR — runs IN PARALLEL with the agents (CPU)"]
        R1["RapidOCR"]
        T1["tesseract (eng)"]
    end

    PRE --> AG
    subgraph AG["③ AI agents (Ollama, GPU) — see §2"]
        A1["fast_scanner"] --> A2["field_extractor"] --> TR["transcriber"] --> A3["accuracy_verifier"] --> A4["layout_analyzer"] --> A5["supervisor"]
    end

    AG --> CON["④ Consensus<br/>majority · specialist · script rule<br/>supervisor pick only if an agent produced it"]
    CON --> INT["⑤ Interpretation of the transcriber's raw text — see §3<br/>names · dates · document number · sex · native-script name"]
    OCR --> INT
    INT --> VER

    subgraph VER["⑥ Verification against evidence — see §5"]
        V1["Barcode fields"] --> V2["MRZ: OCR text → agents' lines → local OCR vote"]
        V2 --> V3["OCR grounding (tesseract in the document's script)"]
        V3 --> V4["National-ID checksums & encoded birth date / sex"]
        V4 --> V5["Names: parts · order · Latin-primary · Ge'ez ↔ Latin check"]
        V5 --> V6["Calendars · plausibility · expiry"]
    end

    OCR --> VER
    VER --> OUT[/"✅ Result<br/>per field: value · confidence · source · verified_by · flags · alternatives<br/>+ human_review_required with reasons"/]

    style AG fill:#eef4ff,stroke:#4a6fd8
    style VER fill:#eefaf0,stroke:#3a9a55
    style INT fill:#fff7e6,stroke:#d49a1a
```

**Two modes.** Both run steps ①, ②, ④, ⑤ and ⑥ in full:

| Mode | Agents run | Typical time |
|---|---|---|
| `fast` | field_extractor + transcriber | ≈ 15–30 s |
| `full` | all six (the five from the prompts file + the transcriber) | ≈ 90–120 s |

---

## 2. How the agents talk to each other

The agents run **one after another**. Each sees the document image(s) plus a
distilled version of what the earlier agents found. None of them talks to
another directly: the orchestrator (`pipeline.run_agents`) builds every message
and merges the answers.

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator<br/>(pipeline.py)
    participant S as fast_scanner<br/>moondream · 2K ctx
    participant E as field_extractor<br/>qwen2.5vl · 8K ctx
    participant T as transcriber<br/>qwen2.5vl · short call
    participant V as accuracy_verifier<br/>llava:13b · 4K ctx
    participant L as layout_analyzer<br/>qwen2.5vl · 8K ctx
    participant U as supervisor<br/>qwen2.5 (text only)

    Note over O: barcode decoded, OCR started in parallel
    O->>S: image + packed prompt (system text folded into user — no system slot)
    S-->>O: type, country, script, orientation, quick fields
    O->>E: image(s) + full prompt + pre-scan hints + barcode ground truth
    E-->>O: 35-field extraction
    O->>T: image(s) + SHORT transcription prompt
    T-->>O: raw text at each label, dates as printed,<br/>number as printed, name in native script
    O->>V: image + packed prompt + previous extraction (subset)
    V-->>O: character-level corrections
    O->>L: image(s) + packed prompt + verified extraction (subset)
    L-->>O: zone-by-zone sweep, missed fields
    O->>U: all four agents' JSON (text only)
    U-->>O: reconciled pick for each field
    Note over O: consensus → interpretation → verification → result
```

What each agent receives and returns:

| # | Agent | Model | Sees | Returns | Mode |
|---|---|---|---|---|---|
| 1 | fast_scanner | moondream | image | document type, country, script, orientation, quick fields | full |
| 2 | field_extractor | qwen2.5vl:7b | image(s) · pre-scan hints · **barcode data as ground truth** | the 35-field schema | fast + full |
| + | **transcriber** | qwen2.5vl:7b | image(s) · short prompt | raw text only: surname / given names at their labels (Latin **and** native script), dates exactly as printed, document number as printed, sex, other labelled numbers | fast + full |
| 3 | accuracy_verifier | llava:13b | image · the extraction so far | corrections of look-alike characters, MRZ check | full |
| 4 | layout_analyzer | qwen2.5vl:7b | image(s) · the verified extraction | fields hidden in stamps, margins, back side | full |
| 5 | supervisor | qwen2.5:7b | the four agents' answers (no image) | one pick per field | full |

**Guard-rails on the conversation:**

- **Supervisor picks are checked.** A value the supervisor picks is accepted only if at least one agent actually read it. A value no agent produced is a hallucination and is rejected in code.
- **Each agent sees only relevant earlier data.** Earlier results are passed as a compact subset (non-empty fields only, and only the fields that agent's pass owns), which keeps small models inside their budget.
- **Agents don't vote for each other.** The later agents see earlier answers, so their agreement isn't independent. Only the scanner and the extractor count as independent readers when the engine decides whether a value is a hallucination.

---

## 3. The transcriber: the model reads, the code decides

Tested on 31 real documents, the agents **read** well but **interpreted** badly:

- surname taken as the last word by position
- `01/02/1984` read day-first on a US card
- the 6-digit CAN taken as the document number
- an Ethiopian-calendar date passed off as Gregorian

Asking for raw text *inside* the 5,000-token extraction prompt didn't work (the
model echoed its own guess), so the transcriber is a **separate, short call**.

```mermaid
flowchart LR
    subgraph READ["Model: transcribe exactly"]
        direction TB
        t1["surname_as_labelled<br/>e.g. '1 DOE'"]
        t2["given_names_as_labelled<br/>e.g. '2 JANE QUINN'"]
        t3["surname / given names / full name<br/>in NATIVE script, e.g. 'ለማ' · 'መሳይ ከበደ'"]
        t4["dates as printed<br/>e.g. '01/02/1984' · '29/5/1987 | 2/6/1995'"]
        t5["document number as printed<br/>e.g. '4d DLN 000012345678 CLASS C'"]
    end

    subgraph DECIDE["Code: interpret by rule (printed.py · names.py · ethiopic.py)"]
        direction TB
        d1["Label numbers decide, not position<br/>ISO 18013: 1 = surname, 2 = given names"]
        d2["Native-script name kept only if the<br/>issuer prints that script; Ge'ez checked<br/>syllable by syllable against the Latin name"]
        d3["Country date convention (US month-first)<br/>dual-calendar pair solved by arithmetic"]
        d4["Labels stripped · per-country format templates<br/>O/0, I/1 look-alikes repaired · CAN rejected"]
    end

    t1 --> d1
    t2 --> d1
    t3 --> d2
    t4 --> d3
    t5 --> d4

    OCRE["OCR physical evidence<br/>'1DOE' · 'ROE,'"] -.outranks.-> d1

    style READ fill:#eef4ff,stroke:#4a6fd8
    style DECIDE fill:#fff7e6,stroke:#d49a1a
```

Examples of the code deciding:

| The model transcribed | The code concluded | Rule |
|---|---|---|
| `01/02/1984` on a US card | 1984-01-02 | US prints month-first |
| `29/5/1987 \| 2/6/1995` (Ethiopian licence) | 1995-02-06 | 29/5/1987 EC converts to 6 Feb 1995, so the Gregorian half is month-first |
| `MR JOHN PAUL` | first JOHN, middle PAUL | honorifics dropped |
| `4d DLN 000012345678 CLASS C` | `000012345678` | label words stripped |
| `CA123450A` (Italy) | `CA12345OA` | template `AA99999AA`: a zero in a letter slot is an O |
| `ረና` next to Latin `HANA` | `ሀና` | ረ spells "re"; ሀ spells "ha" |

---

## 4. Solving the token limit of small models

Three of the models have small context windows. llava is 4,096 tokens and
moondream only 2,048, and part of that goes to the image and to the answer.
Rather than truncating the prompts to a fixed "compact" version, the engine
**measures the real budget of every call and packs it with the most relevant
knowledge.**

### 4.1 The budget of one call

```mermaid
flowchart LR
    CTX["Context limit<br/>read from Ollama<br/>(capped at 8K)"] --> MINUS1(("−"))
    IMGT["Image tokens<br/>probed once per model<br/>llava 577 · moondream 729<br/>qwen2.5vl: 1 per 28×28 px"] --> MINUS1
    MINUS1 --> MINUS2(("−"))
    RES["Output reserve<br/>26 × fields + 48"] --> MINUS2
    MINUS2 --> MINUS3(("−"))
    MAR["Safety margin<br/>64"] --> MINUS3
    MINUS3 --> BUD["= TEXT BUDGET<br/>for system + task prompt"]
    style BUD fill:#eefaf0,stroke:#3a9a55,stroke-width:2px
```

Worked numbers (image counts as measured on this Mac):

| Call | Context | Image | Output reserve | Margin | **Text budget** |
|---|---:|---:|---:|---:|---:|
| fast_scanner · moondream (14 fields) | 2,048 | 729 | 412 | 64 | **843** |
| accuracy_verifier · llava:13b (35 fields) | 4,096 | 577 | 958 | 64 | **2,497** |
| field_extractor · qwen2.5vl (35 fields, 1600×1000 image) | 8,192 | 2,090 | 958 | 64 | **5,080** |
| supervisor · qwen2.5 (37 fields, no image) | 8,192 | 0 | 1,010 | 64 | **7,118** |

The system prompt alone is about 1,800 tokens, so it can't fit moondream's 843.
That's why packing is needed.

### 4.2 Packing the budget with the most relevant knowledge

```mermaid
flowchart TD
    P["Prompts from universal_id_extraction_prompts.py"] --> SPLIT["Split into blocks at natural seams<br/>═══ sections · paragraphs · list items<br/>(e.g. 'Ethiopic: Amharic, Tigrinya' is its own block)"]
    SPLIT --> MAND["Mandatory blocks always in<br/>identity · extraction rules · output format · placeholders · return line"]
    SPLIT --> SCORE["Score every other block FOR THIS DOCUMENT"]
    SCORE --> S1["+ fields this call asks for"]
    SCORE --> S2["+ detected script / country<br/>(Ethiopic on an Ethiopian ID, − Cyrillic)"]
    SCORE --> S3["+ document type · topics<br/>(MRZ, address, calendars, names)"]
    S1 & S2 & S3 --> FILL["Fill the budget in score order"]
    MAND --> FILL
    FILL --> CHK{"Estimate > 85%<br/>of the budget?"}
    CHK -- no --> SEND["Send"]
    CHK -- yes --> EXACT["Exact token count through Ollama"]
    EXACT --> FIT{"Fits?"}
    FIT -- yes --> SEND
    FIT -- no --> DROP["Drop the lowest-scored block"] --> EXACT
    SEND --> SYS{"Does the model's chat template<br/>have a system slot?"}
    SYS -- "yes (qwen, llava)" --> MSG1["system + user messages"]
    SYS -- "no (moondream)" --> MSG2["system text folded into the user message<br/>(otherwise Ollama silently drops it)"]

    style FILL fill:#eefaf0,stroke:#3a9a55
    style MSG2 fill:#fff0f0,stroke:#d05050
```

### 4.3 One model, several passes

When everything relevant doesn't fit in one call, the same model can run
**2–3 times**, each pass owning a group of fields and carrying the knowledge
for those fields. The results are merged, so the model behaves like one with a
larger context.

```mermaid
flowchart LR
    SCHEMA["35-field schema"] --> G1["Pass 1 · identity<br/>names · birth date · sex · nationality"]
    SCHEMA --> G2["Pass 2 · document + location<br/>number · dates · issuer · address"]
    SCHEMA --> G3["Pass 3 · machine-readable + metadata<br/>MRZ · barcode · script · calendar"]
    G1 --> K1["+ name & script rules<br/>+ calendar rules"]
    G2 --> K2["+ document-type rules<br/>+ address & MRZ rules"]
    G3 --> K3["+ MRZ rules<br/>+ language rules"]
    K1 & K2 & K3 --> MERGE["Merge — passes own disjoint fields"]
    MERGE --> ONE["One schema-shaped answer"]
```

**How `auto` chooses** (`agent_passes=auto|1|2|3`):

```mermaid
flowchart TD
    A["Plan 1 fully packed call"] --> B{"All relevant knowledge fits?"}
    B -- yes --> ONE["Use 1 call"]
    B -- no --> C["Plan 2, then 3 passes"]
    C --> D{"Every pass gets ≥ 80%<br/>of its relevant knowledge?"}
    D -- yes --> MULTI["Use that many passes"]
    D -- no --> ONE2["Use 1 fully packed call<br/>(measured: moondream split 3× at ~30%<br/>coverage read FEWER fields)"]
```

With the current prompts, `auto` makes one fully packed call per agent:

- llava now gets its complete relevant prompt, about 3,050 of 4,096 tokens (it used to get about 1,960).
- moondream now actually receives its instructions (they were being dropped).

Forcing 2 or 3 passes is still available for longer custom prompts.

**Other measures that keep the budgets real:**

- **Context sizes are fixed.** Probes and real calls use the same context size, because Ollama reloads a model whenever the size changes.
- **Large models are capped at 8K.** At 16K, qwen2.5vl grew to 7.8 GB and the models stopped fitting together in 17.8 GB of GPU memory.
- **Images are shrunk before encoding.** They're EXIF-rotated and scaled down to 1,600 px, so a 12 MP phone photo doesn't waste thousands of vision tokens.
- **Answers are JSON-schema-constrained.** Every answer has exactly the keys asked for, with no prose and no markdown.

---

## 5. Evidence precedence: who wins a disagreement

```mermaid
flowchart TB
    L1["🥇 Barcode / QR — error-corrected by the symbology"]
    L2["🥈 MRZ check digits · national-ID checksums<br/>per field: only what a check digit protects"]
    L3["🥉 Physical label evidence in OCR<br/>'1DOE' · 'ROE,' · calendar pairs · format templates"]
    L4["Transcriber's raw text, interpreted by rule"]
    L5["Agent consensus (vision models) — max confidence 0.85"]
    L1 --> L2 --> L3 --> L4 --> L5
    L5 -. "a lower level never silently overwrites a higher one;<br/>a disagreement on a critical field → human_review_required" .-> L1

    style L1 fill:#e8f5e9,stroke:#2e7d32
    style L2 fill:#f1f8e9,stroke:#558b2f
    style L3 fill:#fffde7,stroke:#9e9d24
    style L4 fill:#fff8e1,stroke:#f9a825
    style L5 fill:#fff3e0,stroke:#ef6c00
```

**Special cases:**

- **MRZ names.** The MRZ's name line has no check digit. The MRZ decides which words are the surname, but it never replaces the visual spelling. When the two OCR engines read the name line differently, a local OCR re-read with per-character voting settles it.
- **MRZ read from OCR.** The MRZ is found straight from the OCR text of every image, including a card's back, without waiting for an agent to transcribe it. Lines from the two OCR engines are combined by role, and the check digits pick the combination that is right.
- **Rejected values are kept.** Every rejected value stays in `alternatives`, so no correction is silent.

---

## 6. From raw readings to name parts

```mermaid
flowchart TD
    IN["Latin + native-script readings<br/>(agents · transcriber · OCR · MRZ)"] --> LAB["Label evidence decides the split<br/>MRZ > OCR numbered/comma labels > transcriber labels > ALL-CAPS surname"]
    LAB --> CLEAN["Clean<br/>honorifics · label words ('NAME') · place of birth · prompt echoes"]
    CLEAN --> PARTS["Name parts (names.py)<br/>surname = one printed field, never split<br/>given names → first word = first_name, rest = middle_name<br/>particles kept together: ABDUL RAHMAN · DE LOS ANGELES"]
    PARTS --> CULT{"No surname labelled?"}
    CULT -- "patronymic (ETH, ERI, SOM)" --> PAT["given · father · grandfather<br/>→ first · middle · last"]
    CULT -- "family-name-first (CHN, KOR, JPN…)" --> FAM["family name first"]
    CULT -- "labelled" --> ORDER
    PAT --> ORDER
    FAM --> ORDER
    ORDER["full_name = first + middle + last<br/>(printed order kept in full_name_as_printed)"]
    ORDER --> LOCAL["Native-script name<br/>kept only if the issuer prints that script"]
    LOCAL --> GEEZ["Ge'ez ↔ Latin check (ethiopic.py)<br/>each word from the reading that spells it best<br/>one-letter fixes: ረና → ሀና for HANA"]
```

---

## 7. Reading the result

Every field in `fields` carries its own evidence trail:

```json
"last_name": {
  "value": "LEMMA",
  "confidence": 0.99,
  "source": "mrz",
  "verified_by": ["mrz", "ocr"],
  "alternatives": ["LEMA"],
  "flags": []
}
```

| Key | Meaning |
|---|---|
| `source` | Where the final value came from: `mrz`, `barcode`, `printed_labels`, `ocr_numbered_labels`, `format_template`, `calendar_arithmetic`, `supervisor`, `consensus`… |
| `verified_by` | Every independent check the value passed |
| `alternatives` | Every value it replaced; nothing is silently discarded |
| `flags` | Why something changed, e.g. `reinterpreted_from_printed`, `corrected_by_latin_spelling`, `dropped_possible_hallucination` |

Document-level keys:

- **`human_review_required`** always comes with plain-language `review_reasons`.
- **`timings_seconds`** breaks each run down by stage.
- **The agent log** shows the number of passes, the knowledge coverage, and planned vs. actual tokens for every call.
