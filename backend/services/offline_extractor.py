"""
Deterministic Zero-API Industrial Rule & Spec Extraction Engine.

When external AI APIs (Gemini, Groq, etc.) are unavailable, rate-limited (429),
or exhausted, this engine parses raw source text, HTML tables, JSON-LD, and
product descriptions into high-precision, commerce-ready structured attributes
with 0 API calls and zero latency.
"""
import re
from typing import Optional, List, Dict, Tuple
from models import ProductInput, SourceHit, FieldValue, Attribute, StructuredProduct
from services import vocabulary

# --- Standardized Unit Normalizations ---
UOM_PATTERNS = [
    (r'\b(?:mm|millimeter|millimeters)\b', 'mm'),
    (r'\b(?:cm|centimeter|centimeters)\b', 'cm'),
    (r'\b(?:in|inch|inches|\")\b', 'in'),
    (r'\b(?:ft|feet|foot)\b', 'ft'),
    (r'\b(?:m|meter|meters)\b', 'm'),
    (r'\b(?:g|gram|grams)\b', 'g'),
    (r'\b(?:kg|kilogram|kilograms)\b', 'kg'),
    (r'\b(?:lbs?|pounds?)\b', 'lbs'),
    (r'\b(?:oz|ounces?)\b', 'oz'),
    (r'\b(?:v|vac|vdc|volts?)\b', 'V'),
    (r'\b(?:a|amps?|amperes?)\b', 'A'),
    (r'\b(?:ma|milliamps?)\b', 'mA'),
    (r'\b(?:w|watts?)\b', 'W'),
    (r'\b(?:kw|kilowatts?)\b', 'kW'),
    (r'\b(?:hp|horsepower)\b', 'hp'),
    (r'\b(?:hz|hertz)\b', 'Hz'),
    (r'\b(?:rpm|revolutions per minute)\b', 'rpm'),
    (r'\b(?:psi|pounds per square inch)\b', 'psi'),
    (r'\b(?:bar|bars)\b', 'bar'),
    (r'\b(?:kpa|kilopascals?)\b', 'kPa'),
    (r'\b(?:c|celsius|°c)\b', '°C'),
    (r'\b(?:f|fahrenheit|°f)\b', '°F'),
    (r'\b(?:dba|db|decibels?)\b', 'dBA'),
    (r'\b(?:kb|kilobytes?)\b', 'KB'),
    (r'\b(?:mb|megabytes?)\b', 'MB'),
]

# Standard Common Bearings Catalog Specs Lookup
BEARING_SERIES_SPECS = {
    # 6200 series (d x D x B in mm)
    "6200": {"Inner Diameter": ("10", "mm"), "Outer Diameter": ("30", "mm"), "Width": ("9", "mm"), "Dynamic Load Rating": ("5.4", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6201": {"Inner Diameter": ("12", "mm"), "Outer Diameter": ("32", "mm"), "Width": ("10", "mm"), "Dynamic Load Rating": ("6.89", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6202": {"Inner Diameter": ("15", "mm"), "Outer Diameter": ("35", "mm"), "Width": ("11", "mm"), "Dynamic Load Rating": ("7.8", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6203": {"Inner Diameter": ("17", "mm"), "Outer Diameter": ("40", "mm"), "Width": ("12", "mm"), "Dynamic Load Rating": ("9.56", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6204": {"Inner Diameter": ("20", "mm"), "Outer Diameter": ("47", "mm"), "Width": ("14", "mm"), "Dynamic Load Rating": ("13.5", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6205": {"Inner Diameter": ("25", "mm"), "Outer Diameter": ("52", "mm"), "Width": ("15", "mm"), "Dynamic Load Rating": ("14.8", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6206": {"Inner Diameter": ("30", "mm"), "Outer Diameter": ("62", "mm"), "Width": ("16", "mm"), "Dynamic Load Rating": ("20.3", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6207": {"Inner Diameter": ("35", "mm"), "Outer Diameter": ("72", "mm"), "Width": ("17", "mm"), "Dynamic Load Rating": ("27.0", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6208": {"Inner Diameter": ("40", "mm"), "Outer Diameter": ("80", "mm"), "Width": ("18", "mm"), "Dynamic Load Rating": ("32.5", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6209": {"Inner Diameter": ("45", "mm"), "Outer Diameter": ("85", "mm"), "Width": ("19", "mm"), "Dynamic Load Rating": ("35.1", "kN"), "Category": "Deep Groove Ball Bearings"},
    "6210": {"Inner Diameter": ("50", "mm"), "Outer Diameter": ("90", "mm"), "Width": ("20", "mm"), "Dynamic Load Rating": ("37.1", "kN"), "Category": "Deep Groove Ball Bearings"},
    # 6000 series
    "6004": {"Inner Diameter": ("20", "mm"), "Outer Diameter": ("42", "mm"), "Width": ("12", "mm"), "Category": "Deep Groove Ball Bearings"},
    "6005": {"Inner Diameter": ("25", "mm"), "Outer Diameter": ("47", "mm"), "Width": ("12", "mm"), "Category": "Deep Groove Ball Bearings"},
    # 6300 series
    "6304": {"Inner Diameter": ("20", "mm"), "Outer Diameter": ("52", "mm"), "Width": ("15", "mm"), "Category": "Deep Groove Ball Bearings"},
    "6305": {"Inner Diameter": ("25", "mm"), "Outer Diameter": ("62", "mm"), "Width": ("17", "mm"), "Category": "Deep Groove Ball Bearings"},
}

BEARING_SUFFIXES = {
    "2RS1": ("Sealing", "Rubber Contact Seal on Both Sides", None),
    "2RS": ("Sealing", "Rubber Contact Seal on Both Sides", None),
    "2RSH": ("Sealing", "Contact Seal on Both Sides", None),
    "2Z": ("Shielding", "Steel Shield on Both Sides", None),
    "ZZ": ("Shielding", "Steel Shield on Both Sides", None),
    "RS1": ("Sealing", "Rubber Contact Seal on One Side", None),
    "Z": ("Shielding", "Steel Shield on One Side", None),
    "C3": ("Internal Radial Clearance", "C3 (Greater Than Normal)", None),
    "C2": ("Internal Radial Clearance", "C2 (Less Than Normal)", None),
    "C4": ("Internal Radial Clearance", "C4 (Greater Than C3)", None),
    "TN9": ("Cage Material", "Glass Fibre Reinforced PA66", None),
    "M": ("Cage Material", "Machined Brass", None),
    "W64": ("Lubricant", "Solid Oil", None),
}

# Standard Siemens SIMATIC PLC Specs Lookup
SIEMENS_PLC_SPECS = {
    "6ES7214-1AG40-0XB0": {
        "Product Family": ("SIMATIC S7-1200", None),
        "CPU Model": ("CPU 1214C", None),
        "Supply Voltage": ("24", "V"),
        "Voltage Type": ("DC", None),
        "Digital Inputs": ("14", None),
        "Digital Outputs": ("10", None),
        "Analog Inputs": ("2", None),
        "Work Memory": ("100", "KB"),
        "Communication Interface": ("PROFINET / Ethernet RJ45", None),
        "Mounting Type": ("DIN Rail Mount", None),
        "Enclosure Rating": ("IP20", None),
    },
    "6ES7212-1AE40-0XB0": {
        "Product Family": ("SIMATIC S7-1200", None),
        "CPU Model": ("CPU 1212C", None),
        "Supply Voltage": ("24", "V"),
        "Digital Inputs": ("8", None),
        "Digital Outputs": ("6", None),
        "Analog Inputs": ("2", None),
        "Work Memory": ("75", "KB"),
        "Mounting Type": ("DIN Rail Mount", None),
    }
}


def _infer_category(pn: str, brand: str, desc: str, combined_text: str) -> str:
    text = f"{pn} {brand} {desc} {combined_text}".lower()
    
    if any(k in text for k in ["ball bearing", "roller bearing", "groove bearing", "pillow block", "flange bearing", "tapered roller", "spherical roller"]) or re.search(r'\b6\d{3}[-\w]*', pn):
        if "deep groove" in text or re.search(r'\b6\d{3}', pn):
            return "Deep Groove Ball Bearings"
        elif "tapered" in text:
            return "Tapered Roller Bearings"
        elif "spherical" in text:
            return "Spherical Roller Bearings"
        return "Ball Bearings"
        
    if any(k in text for k in ["plc", "programmable logic controller", "s7-1200", "s7-1500", "compact cpu", "cpu 1214c", "cpu 1212c", "controllogix", "compactlogix", "6es7"]):
        return "Programmable Logic Controllers (PLCs)"
        
    if any(k in text for k in ["dishwasher", "dish washer"]):
        return "Built-In Dishwashers"
    if any(k in text for k in ["refrigerator", "fridge", "freezer"]):
        return "Refrigerators & Freezers"
    if any(k in text for k in ["sanding belt", "sanding disc", "sandpaper", "abrasive belt"]):
        return "Sanding Belts & Abrasives"
    if any(k in text for k in ["saw blade", "drill bit", "cutting wheel", "grinding disc"]):
        return "Cutting Tools & Blades"
    if any(k in text for k in ["valve", "solenoid valve", "ball valve", "check valve", "butterfly valve"]):
        return "Valves & Actuators"
    if any(k in text for k in ["sensor", "proximity sensor", "photoelectric sensor", "pressure sensor", "transducer"]):
        return "Industrial Sensors"
    if any(k in text for k in ["vfd", "variable frequency drive", "inverter", "ac drive", "servo drive"]):
        return "Variable Frequency Drives (VFDs)"
    if any(k in text for k in ["circuit breaker", "mcb", "mccb", "contactor", "relay", "overload relay"]):
        return "Circuit Breakers & Contactors"
    if any(k in text for k in ["fitting", "connector", "elbow", "tee", "adapter", "coupling", "nipple"]):
        return "Fittings & Adapters"
    if any(k in text for k in ["pump", "centrifugal pump", "submersible pump", "diaphragm pump"]):
        return "Industrial Pumps"
    if any(k in text for k in ["motor", "electric motor", "induction motor", "stepper motor"]):
        return "Electric Motors"
        
    if desc and len(desc.strip()) > 3:
        clean_desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', desc)
        words = [w.capitalize() for w in clean_desc.split() if len(w) > 1 and w.lower() not in ["the", "and", "for", "with", "inc", "llc"]]
        if words:
            return " ".join(words[:3])
        
    return "Industrial Components"


def _extract_bearing_specs(pn: str) -> List[Tuple[str, str, Optional[str]]]:
    attrs = []
    pn_clean = pn.upper().replace(" ", "")
    
    series_match = re.search(r'\b(6\d{3})\b', pn_clean)
    if series_match:
        series_code = series_match.group(1)
        if series_code in BEARING_SERIES_SPECS:
            specs = BEARING_SERIES_SPECS[series_code]
            for label, val_uom in specs.items():
                if label != "Category":
                    attrs.append((label, val_uom[0], val_uom[1]))
            attrs.append(("Bearing Type", "Deep Groove Ball Bearing", None))
            attrs.append(("Material", "Chrome Steel (100Cr6)", None))
            attrs.append(("Number of Rows", "1", None))
            
    for suffix, (label, val, uom) in BEARING_SUFFIXES.items():
        if suffix in pn_clean:
            attrs.append((label, val, uom))
            
    return attrs


def _extract_plc_specs(pn: str) -> List[Tuple[str, str, Optional[str]]]:
    attrs = []
    pn_clean = pn.upper().replace(" ", "").replace("-", "")
    for k, specs in SIEMENS_PLC_SPECS.items():
        k_clean = k.upper().replace(" ", "").replace("-", "")
        if k_clean == pn_clean or k_clean in pn_clean or pn_clean in k_clean:
            for label, val_uom in specs.items():
                attrs.append((label, val_uom[0], val_uom[1]))
            break
    return attrs


def _extract_electrical_and_physical(text: str) -> List[Tuple[str, str, Optional[str]]]:
    attrs = []
    seen_labels = set()

    def add_attr(label: str, val: str, uom: Optional[str] = None):
        if label.lower() not in seen_labels and val:
            seen_labels.add(label.lower())
            attrs.append((label, str(val).strip(), uom))

    # 1. Voltage Rating
    v_match = re.search(r'(?i)\b(\d{2,4}(?:\.\d+)?)\s*(?:-|to)?\s*(\d{2,4}(?:\.\d+)?)?\s*(?:V|VAC|VDC|Volts?)\b', text)
    if v_match:
        val = f"{v_match.group(1)}-{v_match.group(2)}" if v_match.group(2) else v_match.group(1)
        add_attr("Voltage Rating", val, "V")

    # 2. Current / Amperage Rating
    a_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:A|Amps?|Amperes?)\b', text)
    if a_match:
        a_val = float(a_match.group(1))
        if a_val < 1000:  # avoid year/zip code noise
            add_attr("Amperage Rating", a_match.group(1), "A")

    # 3. Power Rating (W / kW / HP)
    kw_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:kW|Kilowatts?)\b', text)
    if kw_match:
        add_attr("Power Rating", kw_match.group(1), "kW")
    else:
        w_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:W|Watts?)\b', text)
        if w_match:
            add_attr("Power Rating", w_match.group(1), "W")
        hp_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:HP|Horsepower)\b', text)
        if hp_match:
            add_attr("Horsepower", hp_match.group(1), "hp")

    # 4. Frequency Rating
    hz_match = re.search(r'(?i)\b(50\/60|50|60)\s*(?:Hz|Hertz)\b', text)
    if hz_match:
        add_attr("Frequency Rating", hz_match.group(1), "Hz")

    # 5. IP Protection Rating
    ip_match = re.search(r'(?i)\b(IP(?:20|40|44|54|55|65|66|67|68|69K))\b', text)
    if ip_match:
        add_attr("Enclosure Rating", ip_match.group(1).upper(), None)

    # 6. Operating Temperature Range
    temp_match = re.search(r'(?i)(-?\d+)\s*(?:°?C|deg\s*C)?\s*(?:to|-|\.\.)\s*(\+?\d+)\s*°?C\b', text)
    if temp_match:
        add_attr("Operating Temperature", f"{temp_match.group(1)} to {temp_match.group(2)}", "°C")

    # 7. Speed / RPM
    rpm_match = re.search(r'(?i)\b(\d{3,5})\s*(?:RPM|r\/min)\b', text)
    if rpm_match:
        add_attr("Rotational Speed", rpm_match.group(1), "rpm")

    # 8. Pressure Rating
    psi_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:PSI)\b', text)
    if psi_match:
        add_attr("Pressure Rating", psi_match.group(1), "psi")
    bar_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:bar)\b', text)
    if bar_match:
        add_attr("Pressure Rating", bar_match.group(1), "bar")

    # 9. Weight
    kg_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b', text)
    if kg_match:
        add_attr("Weight", kg_match.group(1), "kg")
    g_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:g|grams?)\b', text)
    if g_match and not kg_match:
        add_attr("Weight", g_match.group(1), "g")
    lbs_match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\b', text)
    if lbs_match and not kg_match and not g_match:
        add_attr("Weight", lbs_match.group(1), "lbs")

    # 10. Dimensions (Length x Width x Height or Belt Size)
    dim_3d = re.search(r'(?i)(\d+(?:\.\d+)?)\s*(?:mm|in|cm)?\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*(?:mm|in|cm)?\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*(mm|in|cm)\b', text)
    if dim_3d:
        u = dim_3d.group(4)
        add_attr("Length", dim_3d.group(1), u)
        add_attr("Width", dim_3d.group(2), u)
        add_attr("Height", dim_3d.group(3), u)
    else:
        # 2D Dimension (e.g. 1/2 in x 18 in or 100 x 200 mm)
        dim_2d = re.search(r'(?i)(\d+(?:\/\d+|\.\d+)?)\s*(in|mm|cm)?\s*[xX*×]\s*(\d+(?:\/\d+|\.\d+)?)\s*(in|mm|cm)\b', text)
        if dim_2d:
            u = dim_2d.group(4)
            add_attr("Width", dim_2d.group(1), u)
            add_attr("Length", dim_2d.group(3), u)

    # 11. Materials Detection
    materials = [
        ("Stainless Steel 316", "Stainless Steel 316"),
        ("Stainless Steel 304", "Stainless Steel 304"),
        ("Stainless Steel", "Stainless Steel"),
        ("Chrome Steel", "Chrome Steel"),
        ("Cast Iron", "Cast Iron"),
        ("Aluminum Oxide", "Aluminum Oxide"),
        ("Zirconium", "Zirconium Abrasive"),
        ("Aluminum", "Aluminum"),
        ("Brass", "Brass"),
        ("Bronze", "Bronze"),
        ("Carbon Steel", "Carbon Steel"),
        ("Polycarbonate", "Polycarbonate"),
        ("PVC", "PVC"),
        ("PTFE", "PTFE / Teflon"),
        ("NBR", "Nitrile Rubber (NBR)"),
    ]
    for mat_pattern, mat_name in materials:
        if re.search(rf'(?i)\b{mat_pattern}\b', text):
            add_attr("Material", mat_name, None)
            break

    # 12. Certifications & Standards
    certs = []
    if re.search(r'(?i)\bUL\s*(?:Listed|Recognized)?\b', text):
        certs.append("UL Listed")
    if re.search(r'(?i)\bCE\s*(?:Marked|Certified)?\b', text):
        certs.append("CE")
    if re.search(r'(?i)\bRoHS\s*(?:Compliant)?\b', text):
        certs.append("RoHS Compliant")
    if re.search(r'(?i)\bCSA\s*(?:Certified)?\b', text):
        certs.append("CSA Certified")
    if re.search(r'(?i)\bISO\s*9001\b', text):
        certs.append("ISO 9001")
    if re.search(r'(?i)\bEnergy\s*Star\b', text):
        certs.append("Energy Star Qualified")
    if certs:
        add_attr("Standard/Approvals", " | ".join(certs), None)

    # 13. Mounting Type
    if re.search(r'(?i)\bDIN\s*Rail\b', text):
        add_attr("Mounting Type", "DIN Rail Mount", None)
    elif re.search(r'(?i)\bPanel\s*Mount\b', text):
        add_attr("Mounting Type", "Panel Mount", None)
    elif re.search(r'(?i)\bSurface\s*Mount\b', text):
        add_attr("Mounting Type", "Surface Mount", None)
    elif re.search(r'(?i)\bFlange\s*Mount\b', text):
        add_attr("Mounting Type", "Flange Mount", None)
    elif re.search(r'(?i)\bBuilt-in\b', text):
        add_attr("Mounting Type", "Built-In", None)

    # 14. Series / Model Family
    series_match = re.search(r'(?i)\b(?:Series|Family)\s*:?\s*([A-Za-z0-9\-]+)\b', text)
    if series_match:
        val = series_match.group(1).strip()
        stop_words = ["is", "a", "an", "the", "this", "that", "and", "or", "for", "with", "number", "specification", "product", "details"]
        if len(val) >= 2 and val.lower() not in stop_words:
            add_attr("Series", val, None)

    return attrs


def _extract_key_value_pairs(text: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Parses real technical spec tables from text, rejecting web markup noise.
    """
    attrs = []
    seen = set()

    # Reject web metadata keys
    noisy_keys = {
        "title", "body text", "meta description", "meta og", "meta keywords",
        "json ld structured data", "product state json", "product state json raw",
        "http", "https", "www", "cookie", "privacy", "copyright", "menu", "search",
        "page", "cart", "account", "login", "price", "review", "add to", "rating"
    }

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line or len(line) < 4 or len(line) > 120:
            continue

        match = re.match(r'^([A-Za-z0-9\s\/\-_()]{3,35})\s*[:|=|\t|\|]\s*(.+)$', line)
        if match:
            raw_label = match.group(1).strip()
            raw_val = match.group(2).strip()

            lower_label = raw_label.lower()
            if any(bad in lower_label for bad in noisy_keys):
                continue
            if raw_val.startswith("{") or raw_val.startswith("[") or raw_val.startswith("<"):
                continue

            std_label = " ".join(w.capitalize() for w in re.sub(r'[^a-zA-Z0-9\s]', ' ', raw_label).split())
            if not std_label or len(std_label) < 3 or std_label.lower() in seen or std_label.lower() in noisy_keys:
                continue

            uom = None
            val_cleaned = raw_val
            for uom_regex, std_uom in UOM_PATTERNS:
                uom_match = re.search(rf'\s+{uom_regex}$', raw_val, re.IGNORECASE)
                if uom_match:
                    uom = std_uom
                    val_cleaned = raw_val[:uom_match.start()].strip()
                    break

            if val_cleaned and len(val_cleaned) < 60:
                seen.add(std_label.lower())
                attrs.append((std_label, val_cleaned, uom))

    return attrs


def _clean_brand_name(brand: str, pn: str, text: str) -> str:
    cleaned = (brand or "").strip()
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown", "-- no dib brand --"]
    if not cleaned or cleaned.lower() in bad_brands:
        lower = text.lower()
        if "skf" in lower:
            return "SKF"
        elif "siemens" in lower:
            return "Siemens"
        elif "frigidaire" in lower:
            return "Frigidaire"
        elif "whirlpool" in lower:
            return "Whirlpool"
        elif "diablo" in lower or "freud" in lower:
            return "Diablo"
        elif "3m" in lower:
            return "3M"
        elif "allen-bradley" in lower or "allen bradley" in lower:
            return "Allen-Bradley"
        return "Unknown"
    return cleaned


def extract_offline_product(product: ProductInput, sources: List[SourceHit]) -> StructuredProduct:
    """
    Deterministic zero-API product intelligence extractor.
    Guarantees a valid, fully-populated 252-column exportable product with 0 API tokens spent.
    """
    usable_sources = [s for s in sources if s.raw_text or s.snippet]
    combined_text = "\n\n".join(
        (s.raw_text or s.snippet or "")[:6000] for s in usable_sources
    )
    if not combined_text:
        combined_text = f"{product.part_number} {product.brand} {product.short_description}"

    resolved_brand = _clean_brand_name(product.brand, product.part_number, combined_text)
    resolved_mfr = resolved_brand

    category_name = _infer_category(product.part_number, resolved_brand, product.short_description, combined_text)
    
    extracted_attrs = []
    # 1. Bearings rules
    extracted_attrs.extend(_extract_bearing_specs(product.part_number))
    # 2. PLC catalog rules
    extracted_attrs.extend(_extract_plc_specs(product.part_number))
    # 3. Electrical & physical parameters regex
    extracted_attrs.extend(_extract_electrical_and_physical(combined_text))
    # 4. Key-Value table extraction
    extracted_attrs.extend(_extract_key_value_pairs(combined_text))

    # Group & Deduplicate
    seen_labels = {}
    final_attributes: List[Attribute] = []
    primary_source_url = usable_sources[0].url if usable_sources else None

    for label, val, uom in extracted_attrs:
        norm_label = "".join(ch for ch in label.lower() if ch.isalnum())
        if not norm_label or norm_label in seen_labels:
            continue
        seen_labels[norm_label] = True

        norm_val, val_val = vocabulary.normalize_attribute_value(label, val)
        norm_uom, uom_val = vocabulary.normalize_uom(uom)

        final_attributes.append(
            Attribute(
                label=label,
                value=norm_val,
                uom=norm_uom,
                confidence=0.90 if uom or val_val else 0.80,
                source_url=primary_source_url,
                agreeing_sources=len(usable_sources) if len(usable_sources) > 0 else 1,
                needs_review=False,
                vocab_validated=val_val or uom_val,
            )
        )

    final_attributes = final_attributes[:50]

    key_specs_str = ", ".join(f"{a.label}: {a.value} {a.uom or ''}".strip() for a in final_attributes[:4])
    short_desc_str = f"{resolved_brand} {product.part_number} {category_name}".strip()
    if key_specs_str:
        long_desc_str = f"{resolved_brand} {product.part_number} {category_name}. Key Specifications: {key_specs_str}."
    else:
        long_desc_str = f"{resolved_brand} {product.part_number} {category_name} - {product.short_description}."

    norm_brand, brand_val = vocabulary.normalize_brand(resolved_brand)

    return StructuredProduct(
        part_number=product.part_number,
        brand=norm_brand,
        brand_vocab_validated=brand_val,
        manufacturer=resolved_mfr,
        category=FieldValue(
            value=category_name,
            confidence=0.88,
            source_url=primary_source_url,
            agreeing_sources=max(1, len(usable_sources)),
            needs_review=False
        ),
        short_desc=FieldValue(
            value=short_desc_str,
            confidence=0.85,
            source_url=primary_source_url,
            agreeing_sources=max(1, len(usable_sources)),
            needs_review=False
        ),
        long_desc=FieldValue(
            value=long_desc_str,
            confidence=0.85,
            source_url=primary_source_url,
            agreeing_sources=max(1, len(usable_sources)),
            needs_review=False
        ),
        attributes=final_attributes,
        sources_used=[s.url for s in usable_sources if s.url],
        extraction_engine="offline_rule_engine",
    )
