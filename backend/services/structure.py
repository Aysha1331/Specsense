"""
STAGE 3 + 4: Structure & Score Confidence (Resilient Multi-Tier Cascade)

Architecture:
1. Mode "offline": 100% deterministic local rule & spec extraction via offline_extractor.py. (0 API calls, zero latency).
2. Mode "auto" / "eco":
   - Tier 1: Gemini (Key pool rotation + auto backoff retry).
   - Tier 2: Groq (Key pool rotation + Qwen/Llama).
   - Tier 3: Local Ollama (http://localhost:11434 if running).
   - Tier 4: Deterministic Industrial Spec & Rule Engine (offline_extractor.py).
   
GUARANTEE: This service NEVER raises an unhandled 500 error due to quota exhaustion,
rate limits (429), or network dropouts.
"""
import os
import json
import time
import asyncio
import httpx
from collections import defaultdict
import google.generativeai as genai
from groq import Groq
from models import ProductInput, SourceHit, StructuredProduct, FieldValue, Attribute
from services import vocabulary
from services.offline_extractor import extract_offline_product

gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

gemini_key_idx = 0
groq_key_idx = 0
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_ATTRIBUTES = 50

SYSTEM_PROMPT = """You are a product data extraction engine for an industrial commerce catalog.
You will be given a product's known info (part number, brand, short description) and text from
MULTIPLE numbered sources (webpages, datasheet PDFs, or provided datasets). Each source may or
may not actually be about this exact product.

For EACH source, independently extract:

1. category: the product's category/classification (e.g. "Deep Groove Ball Bearing", "Programmable Logic Controller")
2. brand: the product's brand name (e.g. "Frigidaire", "Whirlpool", "Diablo", "3M", "Milwaukee"). Clean it from distributor prefixes/suffixes. Only report if stated. CRITICAL: Never use distributor/buying group/cooperative names like "Appliance Dealers Cooperative", "APPDE", "Jam Industrial Supply", or generic placeholders like "-- Unbranded --", "-- No Unilog Brand --" as the brand. Look for the true product brand name.
3. manufacturer: the product's manufacturer company name (e.g. "Rheem Manufacturing", "Whirlpool Corporation", "Freud Inc"). Only report if stated. CRITICAL: Never use distributor/buying group/cooperative names like "Appliance Dealers Cooperative", "APPDE", "Jam Industrial Supply", or generic placeholders. Look for the true manufacturing company.
4. short_desc: a concise ~10-15 word description suitable for a mobile listing
5. long_desc: a fuller 1-3 sentence description with key specs included
6. attributes: EVERY distinct technical attribute this source states about the product --
   do not limit yourself to a fixed list. Use clear, standardized attribute names in Title Case
   (e.g. "Voltage Rating" not "voltage", "Mounting Type" not "mount", "Sound Level" not "noise").
   For each attribute give: label (short standardized name), value (the value, no units embedded),
   and uom (unit of measure, e.g. "mm", "g", "V", "kg" -- or null if the value has no unit, like
   a certification name).

Only report what THIS source's text directly states -- do not guess, do not use outside
knowledge, do not invent attributes not actually present in the text. It is completely normal
and expected for most fields to end up empty -- real commerce catalogs leave the large majority
of possible attributes blank when a source doesn't state them, rather than guessing.

EXAMPLE:
  category: "Built-In Dishwashers"
  brand: "Whirlpool"
  manufacturer: "Whirlpool Corporation"
  short_desc: "Whirlpool Eco Series WDTS7024RZ Dishwasher, Built-in Mounting, Stainless Steel"
  long_desc: "Whirlpool dishwasher, Eco Series, 120V, 10A, built-in mounting, 41 dBA sound level, stainless steel."
  attributes: [
    {"label": "Series", "value": "Eco Series", "uom": null},
    {"label": "Voltage Rating", "value": "120", "uom": "V"},
    {"label": "Amperage Rating", "value": "10", "uom": "A"},
    {"label": "Mounting Type", "value": "Built-in", "uom": null},
    {"label": "Sound Level", "value": "41", "uom": "dBA"},
    {"label": "Material", "value": "Stainless Steel", "uom": null}
  ]

Respond ONLY with valid JSON in this exact shape, no other text, no markdown fences:
{
  "sources": [
    {
      "source_index": 0,
      "category": "...",
      "brand": "...",
      "manufacturer": "...",
      "short_desc": "...",
      "long_desc": "...",
      "attributes": [
        {"label": "Material", "value": "Chrome Steel", "uom": null},
        {"label": "Weight", "value": "106", "uom": "g"}
      ]
    }
  ]
}
"""


def get_provider_status() -> dict:
    """Reports the operational status of all AI & offline extraction providers."""
    return {
        "gemini": {
            "configured": len(gemini_keys) > 0,
            "key_count": len(gemini_keys),
            "model": "gemini-3.6-flash / gemini-3.5-flash",
        },
        "groq": {
            "configured": len(groq_keys) > 0,
            "key_count": len(groq_keys),
            "model": GROQ_MODEL,
        },
        "ollama": {
            "endpoint": OLLAMA_URL,
            "model": OLLAMA_MODEL,
        },
        "offline_rule_engine": {
            "status": "ready",
            "capabilities": ["Bearing Taxonomy", "Spec Table Parser", "Electrical/Physical Regex", "Vocabulary Normalizer"],
            "api_cost": "$0.00 (Zero API Calls)",
        }
    }


def _empty_extraction(num_sources: int) -> dict:
    return {"sources": [{"source_index": i, "category": None, "short_desc": None, "long_desc": None, "attributes": []} for i in range(num_sources)]}


def _call_gemini(user_prompt: str) -> dict:
    global gemini_key_idx
    if not gemini_keys:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    
    current_key = gemini_keys[gemini_key_idx % len(gemini_keys)]
    genai.configure(api_key=current_key)
    
    # Try latest models in cascade
    for m_name in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(
                user_prompt,
                generation_config={"temperature": 0, "max_output_tokens": 4096, "response_mime_type": "application/json"},
                request_options={"timeout": 6.0}
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            return _parse_json_loosely(raw)
        except Exception as me:
            if "404" in str(me) or "not found" in str(me).lower():
                continue
            raise me
    raise RuntimeError("No available Gemini model responded successfully.")


def _call_groq(user_prompt: str) -> dict:
    global groq_key_idx
    if not groq_keys:
        raise RuntimeError("GROQ_API_KEY not configured.")
    
    current_key = groq_keys[groq_key_idx % len(groq_keys)]
    client = Groq(api_key=current_key)
    
    extra_body = {}
    if "qwen" in GROQ_MODEL.lower():
        extra_body["reasoning_effort"] = "none"
    elif "gpt-oss" in GROQ_MODEL.lower():
        extra_body["reasoning_format"] = "hidden"

    for m in [GROQ_MODEL, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0,
                max_tokens=4096,
                timeout=4.0,
                extra_body=extra_body if "qwen" in m.lower() else {}
            )
            raw = response.choices[0].message.content.strip()
            return _parse_json_loosely(raw)
        except Exception as ge:
            if "model_not_found" in str(ge).lower() or "deprecated" in str(ge).lower():
                continue
            raise ge
    raise RuntimeError("No available Groq model responded successfully.")


def _call_ollama(user_prompt: str) -> dict:
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": user_prompt, "stream": False, "format": "json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("response", "")
                return _parse_json_loosely(raw)
    except Exception as e:
        raise RuntimeError(f"Ollama local endpoint unavailable: {e}")
    raise RuntimeError("Ollama returned invalid status")


def _close_truncated_json(s: str) -> str:
    s = s.strip()
    if s.endswith(","):
        s = s[:-1].strip()
        
    stack = []
    in_string = False
    escape = False
    
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
                    
    if in_string:
        s += '"'
        
    while stack:
        close_ch = stack.pop()
        s = s.strip()
        if s.endswith(","):
            s = s[:-1].strip()
        s += close_ch
        
    return s


def _parse_json_loosely(raw: str) -> dict:
    import re
    cleaned = raw.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    start = cleaned.find("{")
    if start != -1:
        cleaned = cleaned[start:]
        
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            repaired = _close_truncated_json(cleaned)
            return json.loads(repaired)
        except Exception:
            raise


def _call_llm_with_fallback(user_prompt: str) -> tuple[dict, str]:
    """
    Cascade through available AI providers.
    Returns (parsed_json_dict, provider_name).
    """
    global gemini_key_idx, groq_key_idx

    # 1. Try Gemini
    if gemini_keys:
        attempts = len(gemini_keys)
        for attempt in range(attempts):
            try:
                res = _call_gemini(user_prompt)
                gemini_key_idx += 1
                return res, "gemini"
            except Exception as e:
                msg = str(e)
                print(f"[structure] Gemini call failed with key {gemini_key_idx % len(gemini_keys)}: {msg}")
                gemini_key_idx += 1
                if ("429" in msg or "quota" in msg.lower()) and attempt < attempts - 1:
                    time.sleep(1.5)
                    continue

    # 2. Try Groq
    if groq_keys:
        attempts = len(groq_keys)
        for attempt in range(attempts):
            try:
                print(f"[structure] Falling back to Groq using key {groq_key_idx % len(groq_keys)}...")
                res = _call_groq(user_prompt)
                groq_key_idx += 1
                return res, "groq"
            except Exception as e:
                print(f"[structure] Groq call failed: {e}")
                groq_key_idx += 1

    # 3. Try Local Ollama (if running)
    try:
        res = _call_ollama(user_prompt)
        print("[structure] Used local Ollama model for extraction.")
        return res, "ollama"
    except Exception:
        pass

    raise RuntimeError("All external AI providers exhausted or unavailable.")


def _clean_source_url(url: str | None) -> str | None:
    if not url:
        return None
    url_lower = url.lower()
    bad_domains = ["bing.com", "duckduckgo.com", "google.com", "deepl.com", "translate.", "apple.com", "itunes", "microsoft.com", "amazon.", "ebay.", "yahoo.com", "cnet.com", "spotify.com"]
    if any(bad in url_lower for bad in bad_domains):
        return None
    return url


def _normalize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


async def structure_product(product: ProductInput, sources: list[SourceHit]) -> StructuredProduct:
    usable_sources = [s for s in sources if s.raw_text or s.snippet]
    mode = (getattr(product, "mode", None) or "auto").lower()

    # If user explicitly requested offline mode, skip all API attempts
    if mode == "offline":
        print(f"[structure] Offline mode active for {product.part_number} -- using deterministic spec engine (0 API calls)")
        res = extract_offline_product(product, sources)
        res.extraction_engine = "offline_rule_engine"
        return res

    # AI Cascade with automatic fallback
    parsed = None
    engine_used = "ai"
    try:
        source_blocks = []
        for i, s in enumerate(usable_sources):
            text = s.raw_text
            if not text and s.snippet:
                text = f"[Web scrape snippet]: {s.snippet}"
            text = (text or "")[:4000]
            clean_url = _clean_source_url(s.url) or "Engineering Reference"
            source_blocks.append(f"--- SOURCE {i} ({s.origin}): {clean_url} ---\n{text}")
        combined = "\n\n".join(source_blocks)

        user_prompt = f"""{SYSTEM_PROMPT}

KNOWN PRODUCT INFO:
Part Number: {product.part_number}
Brand: {product.brand}
Short Description: {product.short_description}

SOURCES:
{combined}
"""
        parsed, engine_used = await asyncio.to_thread(_call_llm_with_fallback, user_prompt)
    except Exception as e:
        print(f"[structure] AI providers exhausted or rate-limited: {e}")
        print("[structure] Seamlessly falling back to Deterministic Offline Spec Engine...")
        res = extract_offline_product(product, sources)
        res.extraction_engine = "offline_rule_engine"
        return res

    if not parsed or not parsed.get("sources"):
        res = extract_offline_product(product, sources)
        res.extraction_engine = "offline_rule_engine"
        return res

    per_source = parsed.get("sources", [])

    # ---- Category, short_desc, long_desc Cross-Validation ----
    def resolve_text_field(field_name: str, freeform: bool) -> FieldValue:
        supporting = []
        for entry in per_source:
            idx = entry.get("source_index")
            value = entry.get(field_name)
            if value and idx is not None and 0 <= idx < len(usable_sources):
                supporting.append((value, usable_sources[idx]))

        if not supporting:
            return FieldValue(value=None, confidence=0.15, source_url=None, agreeing_sources=0, needs_review=True)

        if freeform:
            supporting.sort(key=lambda pair: len(pair[0]), reverse=True)
            chosen_value, chosen_source = supporting[0]
            return FieldValue(
                value=chosen_value,
                confidence=0.85 if len(supporting) >= 2 else 0.70,
                source_url=_clean_source_url(chosen_source.url),
                agreeing_sources=len(supporting),
                needs_review=False,
            )

        groups = defaultdict(list)
        for value, src in supporting:
            groups[value.strip().lower()].append((value, src))
        best_key = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_key]
        agreeing_count = len(best_group)
        chosen_value, chosen_source = best_group[0]
        conflicting = len(groups) > 1

        if agreeing_count >= 2 and not conflicting:
            confidence, needs_review = 0.95, False
        elif agreeing_count >= 2 and conflicting:
            confidence, needs_review = 0.75, True
        elif conflicting:
            confidence, needs_review = 0.40, True
        else:
            confidence, needs_review = 0.70, False

        return FieldValue(
            value=chosen_value, confidence=confidence, source_url=_clean_source_url(chosen_source.url),
            agreeing_sources=agreeing_count, needs_review=needs_review,
        )

    category = resolve_text_field("category", freeform=False)
    extracted_brand_fv = resolve_text_field("brand", freeform=False)
    extracted_mfr_fv = resolve_text_field("manufacturer", freeform=False)
    
    resolved_brand = extracted_brand_fv.value if extracted_brand_fv.value else product.brand
    resolved_mfr = extracted_mfr_fv.value if extracted_mfr_fv.value else resolved_brand

    # Clean distributor names
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown", "-- no dib brand --"]
    if not resolved_brand or resolved_brand.lower().strip() in bad_brands:
        desc_lower = (product.short_description or "").lower()
        if "frigidaire" in desc_lower:
            resolved_brand = "Frigidaire"
        elif "whirlpool" in desc_lower:
            resolved_brand = "Whirlpool"
        elif "kitchenaid" in desc_lower:
            resolved_brand = "KitchenAid"
        elif "skf" in desc_lower or "skf" in product.part_number.lower():
            resolved_brand = "SKF"
        elif "siemens" in desc_lower or "6es7" in product.part_number.lower():
            resolved_brand = "Siemens"

    if not resolved_mfr or resolved_mfr.lower().strip() in bad_brands or resolved_mfr == resolved_brand:
        resolved_mfr = resolved_brand

    short_desc = resolve_text_field("short_desc", freeform=True)
    long_desc = resolve_text_field("long_desc", freeform=True)

    # If category or descriptions are missing from LLM, fill from offline heuristic
    if not category.value:
        offline_fallback = extract_offline_product(product, usable_sources)
        category = offline_fallback.category
    if not short_desc.value:
        offline_fallback = extract_offline_product(product, usable_sources)
        short_desc = offline_fallback.short_desc
    if not long_desc.value:
        offline_fallback = extract_offline_product(product, usable_sources)
        long_desc = offline_fallback.long_desc

    # ---- Attributes Grouping ----
    attr_groups = defaultdict(list)
    for entry in per_source:
        idx = entry.get("source_index")
        if idx is None or not (0 <= idx < len(usable_sources)):
            continue
        src = usable_sources[idx]
        for attr in entry.get("attributes", []) or []:
            label = (attr.get("label") or "").strip()
            value = (attr.get("value") or "").strip()
            uom = attr.get("uom")
            if not label or not value:
                continue
            attr_groups[_normalize_label(label)].append((label, value, uom, src))

    final_attributes = []
    for norm_label, entries in attr_groups.items():
        value_groups = defaultdict(list)
        for label, value, uom, src in entries:
            value_groups[value.strip().lower()].append((label, value, uom, src))
        best_key = max(value_groups, key=lambda k: len(value_groups[k]))
        best_group = value_groups[best_key]
        agreeing_count = len(best_group)
        label, value, uom, src = best_group[0]
        conflicting = len(value_groups) > 1

        if agreeing_count >= 2 and not conflicting:
            confidence, needs_review = 0.95, False
        elif agreeing_count >= 2 and conflicting:
            confidence, needs_review = 0.70, True
        elif conflicting:
            confidence, needs_review = 0.35, True
        else:
            confidence, needs_review = 0.65, False

        final_attributes.append(Attribute(
            label=label, value=value, uom=uom, confidence=confidence,
            source_url=_clean_source_url(src.url), agreeing_sources=agreeing_count, needs_review=needs_review,
        ))

    # If LLM returned 0 attributes, supplement with offline spec extractor
    if not final_attributes:
        offline_fallback = extract_offline_product(product, usable_sources)
        final_attributes = offline_fallback.attributes

    final_attributes.sort(key=lambda a: a.confidence, reverse=True)
    final_attributes = final_attributes[:MAX_ATTRIBUTES]

    normalized_brand, brand_validated = vocabulary.normalize_brand(resolved_brand)
    for attr in final_attributes:
        normalized_value, value_validated = vocabulary.normalize_attribute_value(attr.label, attr.value)
        normalized_uom, uom_validated = vocabulary.normalize_uom(attr.uom)
        attr.value = normalized_value
        attr.uom = normalized_uom
        attr.vocab_validated = value_validated or uom_validated

    return StructuredProduct(
        part_number=product.part_number,
        brand=normalized_brand,
        brand_vocab_validated=brand_validated,
        manufacturer=resolved_mfr,
        category=category,
        short_desc=short_desc,
        long_desc=long_desc,
        attributes=final_attributes,
        sources_used=[s.url for s in usable_sources if s.url],
        extraction_engine=engine_used,
    )
