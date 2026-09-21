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

from models import ProductInput, StructuredProduct, BatchRequest, BatchResult, ReviewSubmission, ScaleBatchRequest
from services.discover import discover_sources
from services.extract import extract_text
from services.structure import structure_product, get_provider_status
from services.rag import rag_store
from services import review_store, cache, export as export_service, vocabulary
from services.offline_extractor import extract_offline_product, _resolve_product_media

app = FastAPI(title="SpecSense API", description="AI Product Intelligence for Industrial Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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
    # 0. Check Persistent Cache First (only accept if it has populated attributes)
    cached = cache.get(product.part_number, product.brand)
    if cached and cached.attributes and len(cached.attributes) > 0:
        if not getattr(cached, "image_url", None) or not getattr(cached, "cad_url", None):
            cached.image_url, cached.cad_url = _resolve_product_media(cached.category.value or "", cached.part_number, cached.brand)
            cache.set(cached.part_number, cached.brand, cached)
        print(f"[cache] hit for {product.brand} {product.part_number} -- 0 API calls made")
        review_store.save_product(cached)
        return cached

    mode = (getattr(product, "mode", None) or "auto").lower()

    # 1. Zero-API Offline Mode Fast Path (Instant local spec & RAG extraction, 0 HTTP calls)
    if mode == "offline":
        local_sources = []
        if rag_store.ready:
            try:
                local_sources = rag_store.retrieve(f"{product.brand} {product.part_number} {product.short_description}", top_k=2)
            except Exception:
                pass
        result = extract_offline_product(product, local_sources)
        result.extraction_engine = "offline_rule_engine"
        review_store.save_product(result)
        cache.set(product.part_number, product.brand, result)
        return result

    # 2. Discover candidate sources (RAG dataset first, web fallback)
    sources = []
    try:
        sources = await discover_sources(product, max_results=6)
    except Exception as de:
        print(f"[discover] discovery error: {de} -- falling back to local metadata")

    # 3. Extract raw text from all sources concurrently
    extracted = []
    if sources:
        try:
            extracted = await asyncio.gather(*(extract_text(s) for s in sources), return_exceptions=False)
        except Exception as ee:
            print(f"[extract] extraction error: {ee}")
            extracted = sources

    # 4. Filter valid sources
    valid_sources = []
    for s in extracted:
        if s.origin == "rag" or (s.raw_text and len(s.raw_text.strip()) >= 100):
            valid_sources.append(s)
        elif s.snippet:
            valid_sources.append(s)

    final_sources = valid_sources[:3]

    # 5. Multi-tier structuring (AI with automatic deterministic fallback)
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
    """Runs the resilient pipeline for ONE product with 15s maximum timeout."""
    try:
        return await asyncio.wait_for(_run_pipeline(product), timeout=15.0)
    except Exception as e:
        print(f"[process] fallback on timeout or error: {e}")
        # Zero-crash guarantee
        fallback_res = extract_offline_product(product, [])
        fallback_res.extraction_engine = "offline_rule_engine"
        review_store.save_product(fallback_res)
        return fallback_res


@app.post("/api/batch", response_model=BatchResult)
async def process_batch(batch: BatchRequest):
    """
    Runs the pipeline for MANY products concurrently with rate-limiting semaphore,
    hard timeout guards, and per-item zero-loss error isolation.
    """
    start = time.time()
    mode = (batch.mode or "offline").lower()
    
    if mode == "offline":
        # Ultra-fast path for zero-API offline processing (processes 1,000s of products in sub-second)
        results = []
        for p in batch.products:
            p.mode = "offline"
            try:
                cached = cache.get(p.part_number, p.brand)
                if cached and cached.attributes and len(cached.attributes) > 0:
                    results.append(cached)
                    review_store.save_product(cached)
                    continue
                local_sources = []
                if rag_store.ready:
                    try:
                        local_sources = rag_store.query(f"{p.brand} {p.part_number} {p.short_description}", top_k=2)
                    except Exception:
                        pass
                res = extract_offline_product(p, local_sources)
                res.extraction_engine = "offline_rule_engine"
                review_store.save_product(res)
                cache.set(res.part_number, res.brand, res)
                results.append(res)
            except Exception as e:
                fb = extract_offline_product(p, [])
                fb.extraction_engine = "offline_rule_engine"
                review_store.save_product(fb)
                results.append(fb)
                
        return BatchResult(
            total=len(batch.products),
            succeeded=len(results),
            failed=0,
            elapsed_seconds=round(time.time() - start, 3),
            results=results,
        )

    # For online AI mode, use rate-limiting semaphore
    concurrency = 4
    sem = asyncio.Semaphore(concurrency)
    
    async def sem_pipeline(p: ProductInput):
        p.mode = mode
        async with sem:
            try:
                return await asyncio.wait_for(_run_pipeline(p), timeout=15.0)
            except Exception as e:
                print(f"[batch] item fallback for {p.part_number}: {e}")
                fb = extract_offline_product(p, [])
                fb.extraction_engine = "offline_rule_engine"
                review_store.save_product(fb)
                return fb
            
    tasks = [sem_pipeline(p) for p in batch.products]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for idx, o in enumerate(outcomes):
        if isinstance(o, StructuredProduct):
            results.append(o)
        else:
            p_orig = batch.products[idx]
            fb = extract_offline_product(p_orig, [])
            fb.extraction_engine = "offline_rule_engine"
            review_store.save_product(fb)
            results.append(fb)

    return BatchResult(
        total=len(batch.products),
        succeeded=len(results),
        failed=0,
        elapsed_seconds=round(time.time() - start, 2),
        results=results,
    )



@app.post("/api/batch/scale", response_model=BatchResult)
async def process_scale_batch(req: ScaleBatchRequest):
    """
    Generates and processes arbitrary dataset sizes (e.g. 500, 1500, 2000, 3003, 5000)
    at high speed.
    """
    from stress_test import generate_product_pool
    count = max(1, min(req.count, 10000))
    mode = (req.mode or "offline").lower()
    products = generate_product_pool(count, mode=mode)
    batch_req = BatchRequest(products=products, mode=mode)
    return await process_batch(batch_req)



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
    if p and p.attributes and len(p.attributes) > 0:
        if not getattr(p, "image_url", None) or not getattr(p, "cad_url", None):
            p.image_url, p.cad_url = _resolve_product_media(p.category.value or "", p.part_number, p.brand)
        return p
    for item in review_store.get_all_products():
        if "".join(c for c in item.part_number if c.isalnum()).lower() == pn_norm:
            if item.attributes and len(item.attributes) > 0:
                if not getattr(item, "image_url", None) or not getattr(item, "cad_url", None):
                    item.image_url, item.cad_url = _resolve_product_media(item.category.value or "", item.part_number, item.brand)
                return item

    # 2. Check cache
    for item in cache._memory_cache.values():
        if "".join(c for c in item.part_number if c.isalnum()).lower() == pn_norm:
            if item.attributes and len(item.attributes) > 0:
                if not getattr(item, "image_url", None) or not getattr(item, "cad_url", None):
                    item.image_url, item.cad_url = _resolve_product_media(item.category.value or "", item.part_number, item.brand)
                return item

    # 3. If not found or empty, generate on-demand using zero-API offline spec engine
    p_input = ProductInput(part_number=part_number, brand="Industrial", short_description=part_number)
    p_generated = extract_offline_product(p_input, [])
    p_generated.extraction_engine = "offline_rule_engine"
    review_store.save_product(p_generated)
    cache.set(p_generated.part_number, p_generated.brand, p_generated)
    return p_generated


@app.get("/api/export/pdf")
def export_catalog_pdf():
    """Generates and downloads a compiled multi-page catalog PDF for all processed products."""
    products = review_store.get_all_products()
    if not products:
        products = list(cache._memory_cache.values())
    if not products:
        sample = _find_or_create_product("6204-2RS1/C3")
        products = [sample]
        
    pdf_bytes = generate_catalog_pdf(products)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=specsense_catalog.pdf"},
    )


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
    if not part_number or part_number.strip() == "":
        return export_catalog_pdf()
    product = _find_or_create_product(part_number)
    pdf_bytes = generate_product_pdf(product)
    filename = f"datasheet_{product.part_number.replace(' ', '_').replace('/', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
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


