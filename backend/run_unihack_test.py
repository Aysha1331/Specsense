import os
import csv
import asyncio
import argparse
import time
from dotenv import load_dotenv

# Load env variables before importing services
load_dotenv()

from models import ProductInput, StructuredProduct
from services.discover import discover_sources
from services.extract import extract_text
from services.structure import structure_product
from services.rag import rag_store
from services import vocabulary, cache
from services.export import export_products_csv, HEADER

async def run_pipeline_for_product(part_number: str, brand: str, short_desc: str, mode: str = "auto") -> StructuredProduct:
    product = ProductInput(
        part_number=part_number,
        brand=brand,
        short_description=short_desc,
        mode=mode
    )
    print(f"\nProcessing: PN={part_number}, Brand={brand} (Mode: {mode})...")
    
    # 0. Check cache
    cached = cache.get(part_number, brand)
    if cached:
        print(f"  - [Cache Hit] Loaded instantly from persistent SQLite cache! (Engine: {cached.extraction_engine})")
        return cached

    # 1. Discover candidate sources
    sources = await discover_sources(product, max_results=6)
    print(f"  - Discovered {len(sources)} candidate sources")
    for s in sources:
        print(f"    * [{s.origin}] {s.url}")
        
    # 2. Extract concurrently
    extracted = await asyncio.gather(*(extract_text(s) for s in sources))
    valid_sources = []
    for s in extracted:
        if s.origin == "rag" or (s.raw_text and len(s.raw_text.strip()) >= 100):
            valid_sources.append(s)
            print(f"    * Extracted {len(s.raw_text)} chars from {s.url}")
        elif s.snippet:
            valid_sources.append(s)
            print(f"    * Using snippet ({len(s.snippet)} chars) for {s.url}")
        else:
            print(f"    * Discarding empty/blocked source: {s.url}")
            
    final_sources = valid_sources[:3]
    print(f"  - Using {len(final_sources)} valid sources for structuring")
            
    # 3. Structure & Score & Vocabulary Validation
    result = await structure_product(product, final_sources)
    cache.set(part_number, brand, result)
    
    print(f"  - Engine: {result.extraction_engine}")
    print(f"  - Category: {result.category.value} (Confidence: {result.category.confidence})")
    print(f"  - Brand matched approved list: {result.brand_vocab_validated} ({result.brand})")
    print(f"  - Attributes Extracted: {len(result.attributes)}")
    for attr in result.attributes[:6]:
        print(f"    * {attr.label} = {attr.value} {attr.uom or ''} (conf: {attr.confidence}, vocab-validated: {attr.vocab_validated})")
        
    return result

def clean_brand(e1_brand: str, part_manuf: str) -> str:
    if e1_brand and e1_brand.strip() and e1_brand.strip() != "-- Unbranded --":
        return e1_brand.strip()
    if part_manuf and part_manuf.strip():
        import re
        brand = part_manuf.strip()
        brand = re.sub(r'\s*\([^)]*\)\s*$', '', brand)
        return brand
    return "Unknown"

async def main():
    parser = argparse.ArgumentParser(description="SpecSense UniHack test runner")
    parser.add_argument("--parts", type=str, help="Comma-separated part numbers to process (runs only these)")
    parser.add_argument("--limit", type=int, default=5, help="Limit total number of products to process if no specific parts given")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "eco", "offline"], help="Pipeline execution mode: auto (AI + fallback), eco, or offline (0 API calls)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear persistent cache before running")
    args = parser.parse_args()
    
    if args.clear_cache:
        cache.clear()
        print("[cache] cleared persistent cache")
    
    # Ingest directory (RAG)
    rag_store.ingest_directory()
    # Load controlled vocabulary reference datasets
    vocabulary.load_all()
    
    input_file = os.path.join(os.path.dirname(__file__), "datasets", "unihack_sample_input.csv")
    output_file = os.path.join(os.path.dirname(__file__), "datasets", "unihack_output.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
        
    target_parts = []
    if args.parts:
        target_parts = [p.strip() for p in args.parts.split(",")]
        
    products_to_process = []
    with open(input_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pn = row.get("Mfg_Part_Num", "").strip()
            desc = row.get("Part_Desc", "").strip()
            brand = clean_brand(row.get("E1_Brand", ""), row.get("Part_Manuf", ""))
            
            if target_parts:
                if pn in target_parts:
                    products_to_process.append((pn, brand, desc))
            else:
                products_to_process.append((pn, brand, desc))
                if len(products_to_process) >= args.limit:
                    break
                    
    print(f"Starting pipeline run for {len(products_to_process)} products in mode '{args.mode}'...")
    start_time = time.time()
    
    results = []
    for pn, brand, desc in products_to_process:
        try:
            res = await run_pipeline_for_product(pn, brand, desc, mode=args.mode)
            results.append(res)
        except Exception as e:
            print(f"Error processing {pn}: {e}")
            
    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.2f} seconds.")
    
    if results:
        csv_text = export_products_csv(results)
        with open(output_file, "w", newline="", encoding="utf-8") as out:
            out.write(csv_text)
            
        print(f"Successfully generated {output_file}")
        with open(output_file, newline="", encoding="utf-8") as check:
            reader = csv.reader(check)
            headers = next(reader)
            print(f"Verification: Output contains {len(headers)} columns (Expected: 252)")
            assert len(headers) == 252, "Column count mismatch!"
            
            row_count = sum(1 for _ in reader)
            print(f"Verification: Output contains {row_count} data rows.")
    else:
        print("No results to export.")

if __name__ == "__main__":
    asyncio.run(main())
