"""
Agent specs built from universal_id_extraction_prompts.py — that file is the
single source of truth for prompt text and model assignment.

Each agent carries the FULL system prompt as its knowledge base: the
multi-pass runner (universal/multipass.py) packs as much of it as each call's
real token budget allows, splitting the work across passes when needed. The
file's SYSTEM_PROMPT_COMPACT is kept as `fallback_system` — the floor used
only if token accounting against Ollama is unavailable.

render() substitutes only the three known placeholders (JSON braces in the
prompts are left alone), and the old wrong TD3 check-digit positions are
corrected if they ever reappear in an edited prompts file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import universal_id_extraction_prompts as P

_PLACEHOLDERS = ("previous_extraction", "verified_extraction", "agent_outputs")

_TD3_POSITION_FIXES = {
    "document_number check digit (position 15 in TD3)": "document_number check digit (line 2, character 10 in TD3)",
    "date_of_birth check digit (position 28 in TD3)":   "date_of_birth check digit (line 2, character 20 in TD3)",
    "expiry_date check digit (position 35 in TD3)":     "expiry_date check digit (line 2, character 28 in TD3)",
    "composite check digit (position 44 in TD3)":       "composite check digit (line 2, character 44 in TD3)",
}

# Images per call, by model family (the name before ":"). Context sizes are
# read from Ollama itself — see universal/tokens.py.
_MAX_IMAGES = {"moondream": 1, "llava": 1, "qwen2.5vl": 2, "qwen2.5": 0}
_DEFAULT_MAX_IMAGES = 1


def render(template: str, **values: str) -> str:
    out = template
    for key in _PLACEHOLDERS:
        out = out.replace("{" + key + "}", values.get(key, "{}"))
    return out


def _fix_td3_positions(text: str) -> str:
    for wrong, right in _TD3_POSITION_FIXES.items():
        text = text.replace(wrong, right)
    return text


def _system_sections(text: str) -> tuple[str, dict[str, str]]:
    parts = re.split(r"\n═+\n(.+?)\n═+\n", text)
    intro = parts[0].strip()
    sections = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    return intro, sections


def compact_system_prompt() -> str:
    if getattr(P, "SYSTEM_PROMPT_COMPACT", ""):
        return P.SYSTEM_PROMPT_COMPACT
    intro, sections = _system_sections(P.SYSTEM_PROMPT)
    pieces = [intro]
    if "YOUR EXTRACTION RULES" in sections:
        pieces.append("EXTRACTION RULES:\n" + sections["YOUR EXTRACTION RULES"])
    if "OUTPUT FORMAT" in sections:
        pieces.append(sections["OUTPUT FORMAT"])
    return "\n\n".join(pieces)


def template_fields(prompt: str) -> tuple[str, ...]:
    """Keys of the JSON template a prompt ends with (`"key": ""` lines)."""
    start = prompt.rfind("{\n")
    return tuple(re.findall(r'"([a-z0-9_]+)"\s*:\s*""', prompt[start:] if start != -1 else prompt))


def _max_images(model: str) -> int:
    return _MAX_IMAGES.get(model.split(":", 1)[0].lower(), _DEFAULT_MAX_IMAGES)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    system: str               # knowledge base the packer draws from
    fallback_system: str      # as configured in the prompts file
    user_template: str
    output_fields: tuple[str, ...]
    max_images: int

    @property
    def is_vision(self) -> bool:
        return self.max_images > 0


EXTRACTION_FIELDS = template_fields(P.FIELD_EXTRACTION_PROMPT)
SUPERVISOR_FIELDS = template_fields(P.SUPERVISOR_PROMPT)
SCANNER_FIELDS = template_fields(P.FAST_SCANNER_PROMPT)

# The verifier and layout prompts say "return the same structure" as the
# extraction they're given, so they share the field extractor's schema.
# "As printed" transcription fields, asked of the two qwen2.5vl readers on top
# of the prompts file's schema. Vision models transcribe labels and text
# reliably but INTERPRET poorly (surname = last token by position; 01/02/1984
# read day-first on a US card). So the model transcribes, and code interprets
# (universal/printed.py): label semantics per language, date conventions per
# country, honorifics, comma formats, check-digit-bearing number formats.
AS_PRINTED_FIELDS = (
    "surname_as_labelled", "given_names_as_labelled",
    "surname_local_script", "given_names_local_script", "full_name_local_script", "sex_as_printed",
    "date_of_birth_as_printed", "issue_date_as_printed", "expiry_date_as_printed",
    "document_number_as_printed", "other_numbers_with_labels",
)

TRANSCRIPTION_SYSTEM = "You transcribe identity documents exactly. Return only JSON."

AS_PRINTED_INSTRUCTIONS = """Transcribe these fields from the identity document EXACTLY as printed (same characters, same order, no reformatting, no date conversion):
- surname_as_labelled: the value at the SURNAME / family-name label. Labels include "1" (numbered driving-licence field 1 — follow the NUMBER, not the position; some cards print field 2 above field 1), "LN", "Surname", "Family name", "Last name", "Nom", "Naam", "Efternamn", "Cognome", "Apellidos", "Nachname", and "Name" on German cards. For "SURNAME, Given names" take the part before the first comma. With no label, an ALL-CAPS word in a mixed-case name ("DOE Jane Mary") is the surname.
- given_names_as_labelled: the value at the GIVEN-NAMES label: "2" (numbered field 2), "FN", "Given names", "First name", "Forenames", "Prénoms", "Voornamen", "Förnamn", "Vornamen", "Nombre", and "Nome" (Italian NOME / NAME means GIVEN name).
- surname_local_script / given_names_local_script: the SAME surname and given names as printed in the document's own non-Latin script (Amharic/Ge'ez, Arabic, Chinese, Japanese, Korean, Cyrillic, Greek, Hebrew, Devanagari, Thai...), character for character, e.g. "ለማ" and "መሳይ ከበደ". Do not transliterate or translate. Empty if the document prints no non-Latin name.
- full_name_local_script: the whole non-Latin-script name exactly as printed, for documents that print one full name instead of separate surname / given names (e.g. "ሀና ታደሰ በቀለ", "خالد عمر حسن", "王芳"). Empty if none is printed.
- sex_as_printed: the value at the sex label ("15 SEX", "Sex", "Sexe", "Geslacht", "Kön", "Sesso"), e.g. "M", "F", "V/F", "M/M".
- date_of_birth_as_printed / issue_date_as_printed / expiry_date_as_printed: exactly as printed, e.g. "01/02/1984", "29 OKT/OCT 1972", "06MAR1994", "12/05/1982 | 20/01/1990" (driving-licence fields 3 / 4a / 4b). If two calendars are printed, give both separated by " | ".
- document_number_as_printed: the document / licence / card number (field 4d on US and Canadian licences, field 5 on EU and UK licences). NOT the 6-digit CAN (card access number), NOT a personal identity number, NOT the DD / document discriminator.
- other_numbers_with_labels: every other number printed, each as "LABEL: value", separated by " | "."""

# Measured on the real-ID batch: appended to the 5,000-token extraction prompt,
# these fields merely echoed the model's positional first/last guess; asked in
# their own short call they came back label-correct. So they're a separate
# "transcriber" pass (pipeline.run_agents) run with the field extractor's model.
TRANSCRIBER_AGENT = "transcriber"

_OUTPUT_FIELDS = {
    "fast_scanner":      SCANNER_FIELDS,
    "field_extractor":   EXTRACTION_FIELDS,
    "accuracy_verifier": EXTRACTION_FIELDS,
    "layout_analyzer":   EXTRACTION_FIELDS,
    "supervisor":        SUPERVISOR_FIELDS,
}

AGENT_ORDER = ("fast_scanner", "field_extractor", "accuracy_verifier", "layout_analyzer", "supervisor")


def build_agents() -> dict[str, AgentSpec]:
    agents = {}
    shared = {P.SYSTEM_PROMPT, compact_system_prompt()}
    for name, cfg in P.get_agent_prompts().items():
        # Agents configured with either UIDX system prompt draw on the full one.
        knowledge = P.SYSTEM_PROMPT if cfg["system"] in shared else cfg["system"]
        agents[name] = AgentSpec(
            name=name,
            model=cfg["model"],
            system=knowledge,
            fallback_system=cfg["system"],
            user_template=_fix_td3_positions(cfg["user"]),
            output_fields=_OUTPUT_FIELDS.get(name) or template_fields(cfg["user"]),
            max_images=_max_images(cfg["model"]),
        )
    return agents


def compact_json(data: dict) -> str:
    """Non-empty fields only, no indentation — keeps small-context models
    (llava, moondream) inside their budget when fed a previous stage."""
    return json.dumps({k: v for k, v in data.items() if v not in ("", None)}, ensure_ascii=False)
