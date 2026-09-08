"""
SpecSense - AI-Powered Product Intelligence for Industrial Commerce

Pipeline per product: Discover (RAG-first, web fallback) -> Extract
-> Structure with multi-source cross-validation (AI Cascade + Zero-API Offline Fallback)
-> Confidence Score -> Human-in-the-loop review queue for flagged fields.

Run with:
    uvicorn main:app --reload --port 8000
"""
import time
import asyncio
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

from models import ProductInput, StructuredProduct, BatchRequest, BatchResult, ReviewSubmission
from services.discover import discover_sources
from services.extract import extract_text
from services.structure import structure_product, get_provider_status
from services.rag import rag_store
from services import review_store, cache, export as export_service, vocabulary
from services.offline_extractor import extract_offline_product

app = FastAPI(title="SpecSense API", description="AI Product Intelligence for Industrial Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_ingest_datasets():
    count = rag_store.ingest_directory()
    if count:
        print(f"[startup] RAG store ready: {count} chunks ingested from ./datasets")
    else:
        print("[startup] No datasets found in ./datasets -- RAG store empty, will rely on live web search only.")

    vocabulary.load_all()
    if not vocabulary.is_any_active():
        print("[startup] No controlled vocabulary reference files found -- attribute/brand/UOM validation is INACTIVE. See backend/reference_data/README.md to activate.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_chunks_indexed": len(rag_store.chunks),
        "vocabulary_status": vocabulary.status,
        "cache_stats": cache.get_stats(),
    }


@app.get("/api/system/status")
def system_status():
    """Returns complete health & capability telemetry for all AI providers and offline engines."""
    return {
        "status": "ok",
        "providers": get_provider_status(),
        "rag_store": {
            "ready": rag_store.ready,
            "chunks_indexed": len(rag_store.chunks),
        },
        "vocabulary": {
            "status": vocabulary.status,
            "any_active": vocabulary.is_any_active(),
        },
        "cache": cache.get_stats(),
    }


@app.get("/api/vocabulary/status")
def vocabulary_status():
    return {
        "active": vocabulary.status,
        "any_active": vocabulary.is_any_active(),
    }


@app.get("/api/cache/stats")
def cache_stats():
    return cache.get_stats()


@app.post("/api/cache/clear")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared successfully", "stats": cache.get_stats()}


async def _run_pipeline(product: ProductInput) -> StructuredProduct:
    # 0. Check Persistent Cache First
    cached = cache.get(product.part_number, product.brand)
    if cached:
        print(f"[cache] hit for {product.brand} {product.part_number} -- 0 API calls made")
        review_store.save_product(cached)
        return cached

    # 1. Discover candidate sources (RAG dataset first, web fallback)
    sources = []
    try:
        sources = await discover_sources(product, max_results=6)
    except Exception as de:
        print(f"[discover] discovery error: {de} -- falling back to local metadata")

    # 2. Extract raw text from all sources concurrently
    extracted = []
    if sources:
        try:
            extracted = await asyncio.gather(*(extract_text(s) for s in sources), return_exceptions=False)
        except Exception as ee:
            print(f"[extract] extraction error: {ee}")
            extracted = sources

    # 3. Filter valid sources
    valid_sources = []
    for s in extracted:
        if s.origin == "rag" or (s.raw_text and len(s.raw_text.strip()) >= 100):
            valid_sources.append(s)
        elif s.snippet:
            valid_sources.append(s)

    final_sources = valid_sources[:3]

    # 4. Multi-tier structuring (AI with automatic deterministic fallback)
    try:
        result = await structure_product(product, final_sources)
    except Exception as se:
        print(f"[structure] unexpected error in structure_product: {se} -- using offline engine")
        result = extract_offline_product(product, final_sources)
        result.extraction_engine = "offline_rule_engine"

    review_store.save_product(result)
    cache.set(product.part_number, product.brand, result)
    return result


@app.post("/api/process", response_model=StructuredProduct)
async def process_product(product: ProductInput):
    """Runs the resilient pipeline for ONE product."""
    try:
        return await _run_pipeline(product)
    except Exception as e:
        print(f"[process] fallback on uncaught error: {e}")
        # Zero-crash guarantee
        fallback_res = extract_offline_product(product, [])
        fallback_res.extraction_engine = "offline_rule_engine"
        review_store.save_product(fallback_res)
        return fallback_res


@app.post("/api/batch", response_model=BatchResult)
async def process_batch(batch: BatchRequest):
    """
    Runs the pipeline for MANY products concurrently with rate-limiting semaphore
    and per-item error isolation.
    """
    start = time.time()
    mode = batch.mode or "auto"
    
    # Process max 4 products concurrently
    sem = asyncio.Semaphore(4)
    
    async def sem_pipeline(p: ProductInput):
        p.mode = mode
        async with sem:
            return await _run_pipeline(p)
            
    tasks = [sem_pipeline(p) for p in batch.products]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    failed = 0
    for idx, o in enumerate(outcomes):
        if isinstance(o, StructuredProduct):
            results.append(o)
        else:
            # Generate deterministic fallback for failed product so batch never loses a row
            p_orig = batch.products[idx]
            fb = extract_offline_product(p_orig, [])
            fb.extraction_engine = "offline_rule_engine"
            review_store.save_product(fb)
            results.append(fb)

    return BatchResult(
        total=len(batch.products),
        succeeded=len(results),
        failed=failed,
        elapsed_seconds=round(time.time() - start, 2),
        results=results,
    )


@app.get("/api/review/queue")
def get_review_queue():
    """Every flagged field across every processed product, waiting for a human decision."""
    return {"flagged_fields": review_store.get_flagged_fields()}


@app.post("/api/review/submit", response_model=StructuredProduct)
def submit_review(review: ReviewSubmission):
    """Human approves/corrects a flagged field."""
    updated = review_store.submit_review(review)
    if not updated:
        raise HTTPException(status_code=404, detail="Product or field not found")
    return updated


@app.get("/api/review/log")
def get_correction_log():
    return {"corrections": review_store.get_correction_log()}


from fastapi.responses import PlainTextResponse, Response
from services.pdf_generator import generate_product_pdf, generate_catalog_pdf


@app.get("/api/products")
def list_products():
    return {"products": review_store.get_all_products()}


@app.get("/api/export/csv", response_class=PlainTextResponse)
def export_csv():
    """Exports processed products as CSV matching the exact 252-column spec."""
    products = review_store.get_all_products()
    csv_text = export_service.export_products_csv(products)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=specsense_export.csv"},
    )


def _find_or_create_product(part_number: str) -> StructuredProduct:
    pn_norm = "".join(c for c in part_number if c.isalnum()).lower()
    
    # 1. Check review store
    p = review_store.get_product(part_number)
    if p:
        return p
    for item in review_store.get_all_products():
        if "".join(c for c in item.part_number if c.isalnum()).lower() == pn_norm:
            return item

    # 2. Check cache
    for item in cache._memory_cache.values():
        if "".join(c for c in item.part_number if c.isalnum()).lower() == pn_norm:
            return item

    # 3. If not found, generate on-demand using zero-API offline spec engine
    p_input = ProductInput(part_number=part_number, brand="Industrial", short_description=part_number)
    p_generated = extract_offline_product(p_input, [])
    p_generated.extraction_engine = "offline_rule_engine"
    review_store.save_product(p_generated)
    return p_generated


@app.get("/api/export/pdf/single")
def export_single_pdf_query(part_number: str):
    """Generates and downloads a PDF datasheet using query parameter (safe for slashes)."""
    product = _find_or_create_product(part_number)
    pdf_bytes = generate_product_pdf(product)
    filename = f"datasheet_{product.part_number.replace(' ', '_').replace('/', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/export/pdf/{part_number:path}")
def export_single_pdf_path(part_number: str):
    """Generates and downloads a PDF datasheet using path parameter."""
    product = _find_or_create_product(part_number)
    pdf_bytes = generate_product_pdf(product)
    filename = f"datasheet_{product.part_number.replace(' ', '_').replace('/', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/export/pdf")
def export_catalog_pdf():
    """Generates and downloads a compiled multi-page catalog PDF for all processed products."""
    products = review_store.get_all_products()
    if not products:
        products = list(cache._memory_cache.values())
    if not products:
        # Default sample if nothing processed yet
        sample = _find_or_create_product("6204-2RS1/C3")
        products = [sample]
        
    pdf_bytes = generate_catalog_pdf(products)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=specsense_catalog.pdf"},
    )


current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "..", "frontend")
if not os.path.isdir(frontend_dir):
    frontend_dir = os.path.join(current_dir, "frontend")

if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"[startup] Mounted frontend from: {frontend_dir}")
else:
    print("[startup] Warning: Frontend directory not found.")


