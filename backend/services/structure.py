"""
STAGE 3 + 4: Structure & Score Confidence (Resilient Multi-Tier Cascade)

Architecture:
1. Mode "auto" / "eco":
   - Tier 1: Google Gemini (gemini-3.6-flash / gemini-3.5-flash / gemini-flash-latest).
   - Tier 2: Groq (llama-3.3-70b-versatile / llama-3.1-8b-instant).
   - Tier 3: Local Ollama (http://localhost:11434 if running).
   - Tier 4: Deterministic Industrial Spec & Rule Engine (offline_extractor.py).
2. Mode "offline":
   - 100% deterministic local rule & spec extraction via offline_extractor.py (0 API calls).

Extracts authentic technical specifications directly from discovered search text and datasheets.
"""
import os
import json
import time
import asyncio
import httpx
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

from models import ProductInput, SourceHit, StructuredProduct, FieldValue, Attribute
from services import vocabulary
from services.offline_extractor import extract_offline_product, _resolve_product_media

gemini_key_idx = 0
groq_key_idx = 0

MAX_ATTRIBUTES = 50

SYSTEM_PROMPT = """You are an expert product intelligence and engineering specification extraction engine for an industrial & technical catalog.
You are given a product's part number, brand, and text/tables from MULTIPLE searched sources (datasheets, product pages, spec tables, and search results).

For EACH source, extract ONLY facts and technical specifications verified in that text:

1. category: The standardized product category (e.g. "Solid State Drives (SSDs)", "Deep Groove Ball Bearings", "Programmable Logic Controllers (PLCs)", "Circular Saw Blades", "Miniature Circuit Breakers").
2. brand: The true product brand name (e.g. "Crucial", "Micron", "SKF", "Siemens", "Diablo", "Western Digital", "Schneider Electric"). Never use distributor or retailer names.
3. manufacturer: The manufacturing company.
4. short_desc: A concise ~10-15 word description highlighting key specifications.
5. long_desc: A detailed 1-3 sentence summary covering core parameters, interfaces, and applications.
6. attributes: EVERY verified technical attribute mentioned in the source (e.g. Capacity, Interface, Sequential Read Speed, Sequential Write Speed, NAND Flash Type, TBW, Dimensions, Voltage, Current, Power Rating, Operating Temperature, Mounting Type, Approvals/Standards, Warranty).
   Each attribute must have:
   - label: Clear, standardized attribute name in Title Case (e.g. "Storage Capacity", "Sequential Read Speed", "Supply Voltage", "Operating Temperature Range").
   - value: The exact numeric or descriptive value (without units embedded).
   - uom: Standard unit of measure (e.g. "MB/s", "GB", "TB", "V", "A", "W", "mm", "°C", "IOPS", "TBW", "rpm", "bar", "psi") or null if unitless.

CRITICAL INSTRUCTIONS:
- Do NOT hallucinate or guess random 400V 3-phase machinery attributes for computer hardware, SSDs, consumer electronics, or hand tools.
- Extract actual parametric numbers (speeds, dimensions, voltages, interfaces, capacities) from the source text.

Respond ONLY with valid JSON in this exact structure:
{
  "sources": [
    {
      "source_index": 0,
      "category": "Solid State Drives (SSDs)",
      "brand": "Crucial",
      "manufacturer": "Micron Technology",
      "short_desc": "Crucial MX500 1TB 3D NAND SATA 2.5-Inch Internal Solid State Drive",
      "long_desc": "Crucial MX500 CT1000MX500SSD1 1TB 2.5-inch 7mm SATA III SSD with speeds up to 560 MB/s read and 510 MB/s write.",
      "attributes": [
        {"label": "Storage Capacity", "value": "1 TB", "uom": null},
        {"label": "Interface Type", "value": "SATA III 6.0 Gb/s", "uom": null},
        {"label": "Form Factor", "value": "2.5-inch (7mm)", "uom": null},
        {"label": "Sequential Read Speed", "value": "560", "uom": "MB/s"},
        {"label": "Sequential Write Speed", "value": "510", "uom": "MB/s"}
      ]
    }
  ]
}
"""


def _get_gemini_keys() -> list[str]:
    raw = os.getenv("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_groq_keys() -> list[str]:
    raw = os.getenv("GROQ_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def get_provider_status() -> dict:
    """Reports the operational status of all AI & offline extraction providers."""
    g_keys = _get_gemini_keys()
    gr_keys = _get_groq_keys()
    return {
        "gemini": {
            "configured": len(g_keys) > 0 and genai is not None,
            "key_count": len(g_keys),
            "model": "gemini-3.6-flash / gemini-3.5-flash",
        },
        "groq": {
            "configured": len(gr_keys) > 0 and Groq is not None,
            "key_count": len(gr_keys),
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        },
        "ollama": {
            "endpoint": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            "model": os.getenv("OLLAMA_MODEL", "llama3"),
        },
        "offline_rule_engine": {
            "status": "ready",
            "capabilities": ["Bearing Taxonomy", "Spec Table Parser", "Electrical/Physical Regex", "Vocabulary Normalizer"],
            "api_cost": "$0.00 (Zero API Calls)",
        }
    }


def _parse_json_loosely(raw: str) -> dict:
    import re
    cleaned = raw.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'(\{[\s\S]*\})', cleaned)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return {"sources": []}


def _call_gemini(user_prompt: str) -> dict:
    global gemini_key_idx
    keys = _get_gemini_keys()
    if not keys or genai is None:
        raise RuntimeError("GEMINI_API_KEY is not configured or google.generativeai not installed.")
    
    current_key = keys[gemini_key_idx % len(keys)]
    genai.configure(api_key=current_key)
    
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-2.5-pro",
    ]
    
    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(
                user_prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 4096, "response_mime_type": "application/json"},
                request_options={"timeout": 8.0}
            )
            if response and response.text:
                return _parse_json_loosely(response.text)
        except Exception as me:
            err_str = str(me).lower()
            if "not found" in err_str or "404" in err_str:
                continue
            if "429" in err_str or "quota" in err_str:
                raise me
    raise RuntimeError("No available Gemini model responded.")


def _call_groq(user_prompt: str) -> dict:
    global groq_key_idx
    keys = _get_groq_keys()
    if not keys or Groq is None:
        raise RuntimeError("GROQ_API_KEY not configured or groq package not installed.")
    
    current_key = keys[groq_key_idx % len(keys)]
    client = Groq(api_key=current_key)
    
    for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.1,
                max_tokens=4096,
                timeout=6.0,
            )
            raw = response.choices[0].message.content.strip()
            return _parse_json_loosely(raw)
        except Exception as ge:
            if "model_not_found" in str(ge).lower():
                continue
            raise ge
    raise RuntimeError("No available Groq model responded.")


def _call_ollama(user_prompt: str) -> dict:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.post(
                ollama_url,
                json={"model": ollama_model, "prompt": user_prompt, "stream": False, "format": "json"}
            )
            if resp.status_code == 200:
                data = resp.json()
                return _parse_json_loosely(data.get("response", ""))
    except Exception as e:
        raise RuntimeError(f"Ollama local endpoint unavailable: {e}")
    raise RuntimeError("Ollama returned invalid status")


def _call_llm_with_fallback(user_prompt: str) -> tuple[dict, str]:
    """Cascade through available AI providers."""
    global gemini_key_idx, groq_key_idx

    # 1. Try Gemini
    keys = _get_gemini_keys()
    if keys and genai is not None:
        attempts = len(keys)
        for attempt in range(attempts):
            try:
                res = _call_gemini(user_prompt)
                gemini_key_idx += 1
                return res, "gemini"
            except Exception as e:
                print(f"[structure] Gemini call failed: {e}")
                gemini_key_idx += 1
                if attempt < attempts - 1:
                    time.sleep(1.0)
                    continue

    # 2. Try Groq
    groq_keys = _get_groq_keys()
    if groq_keys and Groq is not None:
        attempts = len(groq_keys)
        for attempt in range(attempts):
            try:
                res = _call_groq(user_prompt)
                groq_key_idx += 1
                return res, "groq"
            except Exception as e:
                print(f"[structure] Groq call failed: {e}")
                groq_key_idx += 1

    # 3. Try Ollama
    try:
        res = _call_ollama(user_prompt)
        return res, "ollama"
    except Exception:
        pass

    raise RuntimeError("All external AI providers exhausted or unavailable.")


def _clean_source_url(url: str | None) -> str | None:
    if not url:
        return None
    url_lower = url.lower()
    bad_domains = ["bing.com", "duckduckgo.com", "google.com", "deepl.com", "translate.", "apple.com", "itunes", "microsoft.com", "amazon.", "ebay.", "yahoo.com", "spotify.com"]
    if any(bad in url_lower for bad in bad_domains):
        return None
    return url


async def structure_product(product: ProductInput, sources: list[SourceHit]) -> StructuredProduct:
    usable_sources = [s for s in sources if s.raw_text or s.snippet]
    mode = (getattr(product, "mode", None) or "auto").lower()

    if mode == "offline":
        res = extract_offline_product(product, sources)
        res.extraction_engine = "offline_rule_engine"
        return res

    parsed = None
    engine_used = "ai"
    try:
        source_blocks = []
        for i, s in enumerate(usable_sources):
            text = s.raw_text
            if not text and s.snippet:
                text = f"[Search Snippet]: {s.snippet}"
            text = (text or "")[:5000]
            clean_url = _clean_source_url(s.url) or s.title or f"Source {i+1}"
            source_blocks.append(f"--- SOURCE {i} ({s.origin}): {clean_url} ---\n{text}")
        combined = "\n\n".join(source_blocks)

        user_prompt = f"""{SYSTEM_PROMPT}

KNOWN PRODUCT INFO:
Part Number: {product.part_number}
Brand: {product.brand}
Short Description: {product.short_description}

SEARCHED SOURCES & DATASHEETS:
{combined if combined.strip() else '[No search text available]'}
"""
        parsed, engine_used = await asyncio.to_thread(_call_llm_with_fallback, user_prompt)
    except Exception as e:
        print(f"[structure] AI cascade fell back to deterministic engine: {e}")
        res = extract_offline_product(product, sources)
        res.extraction_engine = "offline_rule_engine"
        return res

    if not parsed or not parsed.get("sources"):
        res = extract_offline_product(product, sources)
        res.extraction_engine = "offline_rule_engine"
        return res

    per_source = parsed.get("sources", [])

    # Cross-validation helper
    def resolve_text_field(field_name: str, freeform: bool) -> FieldValue:
        supporting = []
        for entry in per_source:
            idx = entry.get("source_index", 0)
            value = entry.get(field_name)
            if value:
                src = usable_sources[idx] if idx < len(usable_sources) else (usable_sources[0] if usable_sources else None)
                supporting.append((value, src))

        if not supporting:
            return FieldValue(value=None, confidence=0.5, source_url=None, agreeing_sources=0, needs_review=False)

        if freeform:
            supporting.sort(key=lambda pair: len(str(pair[0])), reverse=True)
            chosen_value, chosen_source = supporting[0]
            return FieldValue(
                value=str(chosen_value),
                confidence=0.92 if len(supporting) >= 2 else 0.85,
                source_url=_clean_source_url(chosen_source.url) if chosen_source else None,
                agreeing_sources=len(supporting),
                needs_review=False,
            )

        groups = defaultdict(list)
        for value, src in supporting:
            groups[str(value).strip().lower()].append((value, src))
        best_key = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_key]
        agreeing_count = len(best_group)
        chosen_value, chosen_source = best_group[0]

        return FieldValue(
            value=str(chosen_value),
            confidence=0.95 if agreeing_count >= 2 else 0.88,
            source_url=_clean_source_url(chosen_source.url) if chosen_source else None,
            agreeing_sources=agreeing_count,
            needs_review=False,
        )

    resolved_category = resolve_text_field("category", freeform=False)
    resolved_short_desc = resolve_text_field("short_desc", freeform=True)
    resolved_long_desc = resolve_text_field("long_desc", freeform=True)

    # Attributes resolution & cross-validation
    all_extracted_attrs: list[Attribute] = []
    seen_labels = {}

    for entry in per_source:
        idx = entry.get("source_index", 0)
        src = usable_sources[idx] if idx < len(usable_sources) else (usable_sources[0] if usable_sources else None)
        src_url = _clean_source_url(src.url) if src else None

        for a in entry.get("attributes", []):
            raw_label = str(a.get("label") or "").strip()
            raw_val = str(a.get("value") or "").strip()
            raw_uom = a.get("uom")

            if not raw_label or not raw_val or raw_val.lower() in ["none", "null", "n/a"]:
                continue

            norm_label_key = "".join(ch for ch in raw_label.lower() if ch.isalnum())
            if not norm_label_key:
                continue

            if norm_label_key in seen_labels:
                existing = seen_labels[norm_label_key]
                existing.agreeing_sources += 1
                existing.confidence = min(0.98, existing.confidence + 0.05)
                continue

            norm_val, val_val = vocabulary.normalize_attribute_value(raw_label, raw_val)
            norm_uom, uom_val = vocabulary.normalize_uom(raw_uom)

            attr = Attribute(
                label=raw_label,
                value=norm_val,
                uom=norm_uom,
                confidence=0.92 if (val_val or uom_val) else 0.88,
                source_url=src_url,
                agreeing_sources=1,
                needs_review=False,
                vocab_validated=val_val or uom_val,
            )
            seen_labels[norm_label_key] = attr
            all_extracted_attrs.append(attr)

    if not all_extracted_attrs:
        # Fallback to offline extraction if AI returned 0 attributes
        offline_res = extract_offline_product(product, sources)
        all_extracted_attrs = offline_res.attributes

    # Resolve Brand and Manufacturer
    brand_val_res = resolve_text_field("brand", freeform=False)
    resolved_brand = brand_val_res.value if (brand_val_res.value and brand_val_res.value.lower() not in ["industrial", "unknown", ""]) else product.brand
    norm_brand, brand_vocab_val = vocabulary.normalize_brand(resolved_brand)

    mfr_val_res = resolve_text_field("manufacturer", freeform=False)
    resolved_mfr = mfr_val_res.value or resolved_brand

    # Media
    cat_val = resolved_category.value or "Industrial Component"
    img_url, cad_url = _resolve_product_media(cat_val, product.part_number, norm_brand)

    return StructuredProduct(
        part_number=product.part_number,
        brand=norm_brand,
        brand_vocab_validated=brand_vocab_val,
        manufacturer=resolved_mfr,
        category=resolved_category,
        short_desc=resolved_short_desc,
        long_desc=resolved_long_desc,
        attributes=all_extracted_attrs[:MAX_ATTRIBUTES],
        sources_used=[s.url for s in usable_sources if s.url],
        image_url=img_url,
        cad_url=cad_url,
        extraction_engine=engine_used,
    )
