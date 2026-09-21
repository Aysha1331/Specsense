"""
STAGE 1: Discover (High-Resilience Multi-Engine Search)

Takes the minimal product input (part number, brand, description) and finds
authentic candidate source material -- checking ingested RAG datasets FIRST,
then querying a resilient multi-engine live web search (DuckDuckGo, Yahoo, Bing,
Google, AllDatasheet, and SerpAPI).

Extracts authentic URLs, page titles, and rich technical snippets.
"""
import os
import re
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv
from models import ProductInput, SourceHit
from services.rag import rag_store

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# Noise & spam domains to exclude
EXCLUDED_DOMAINS = [
    "youtube.com", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "tiktok.com", "pinterest.com", "reddit.com", "quora.com", "medium.com",
    "deepl.com", "translate.google.", "bing.com/translator", "reverso.net",
    "dictionary.", "thesaurus.", "wiktionary.", "cambridge.org",
    "merriam-webster.", "collinsdictionary.", "vocabulary.com",
    "netflix.com", "spotify.com", "imdb.com", "yelp.com", "tripadvisor.com",
    "aliexpress.", "alibaba.", "temu.com", "shein.com"
]

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1"
}


def _clean_url(url: str) -> str:
    if not url:
        return url
    url = url.replace(r'\u0026', '&').replace(r'\u003d', '=').replace(r'\u003f', '?').replace(r'\u002f', '/')
    try:
        import codecs
        url = codecs.decode(url.encode(), 'unicode-escape').decode('utf-8')
    except Exception:
        pass
    return url.strip()


def _is_excluded_source(url: str) -> bool:
    if not url:
        return True
    url_lower = url.lower()
    
    # Filter search engine internal trackers
    if any(k in url_lower for k in ["bing.com/ck", "duckduckgo.com/l/?", "google.com/url?", "r.search.yahoo.com/_ylt"]):
        return False  # will be decoded to destination
        
    for d in EXCLUDED_DOMAINS:
        if d in url_lower:
            return True

    adult_keywords = ["porn", "xxx", "adult", "sex", "redtube", "pornhub", "xnxx", "xvideos"]
    if any(kw in url_lower for kw in adult_keywords):
        return True

    return False


def _decode_redirect_url(url: str) -> str:
    """Decodes search engine tracking redirects (Bing, DDG, Yahoo, Google) to raw target URLs."""
    if not url:
        return ""
    try:
        # 1. DuckDuckGo redirect: /l/?uddg=https%3A%2F%2F...
        if "duckduckgo.com/l/?" in url or "/l/?uddg=" in url:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                return unquote(qs["uddg"][0])

        # 2. Yahoo redirect: .../RU=https%3a%2f%2f.../RK=2...
        if "r.search.yahoo.com" in url or "/RU=" in url:
            m = re.search(r'/RU=([^/]+)/', url)
            if m:
                return unquote(m.group(1))

        # 3. Google redirect: /url?q=https://...
        if "google.com/url?" in url:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "q" in qs:
                return qs["q"][0]
            if "url" in qs:
                return qs["url"][0]

        # 4. Bing redirect: bing.com/ck/a?!...&u=a1aHR0cHM...
        if "bing.com/ck/a?!" in url:
            import base64
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "u" in qs:
                u_val = qs["u"][0]
                if len(u_val) > 2:
                    encoded = u_val[2:]
                    padding = len(encoded) % 4
                    if padding:
                        encoded += "=" * (4 - padding)
                    return base64.b64decode(encoded).decode("utf-8", errors="ignore")
    except Exception:
        pass
    return _clean_url(url)


async def _ddg_search(query: str, max_results: int) -> list[SourceHit]:
    """DuckDuckGo HTML Search POST method."""
    url = "https://html.duckduckgo.com/html/"
    hits = []
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            resp = await client.post(url, data={"q": query}, headers=BROWSER_HEADERS)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for res in soup.select(".result"):
                    t_el = res.select_one(".result__title")
                    s_el = res.select_one(".result__snippet")
                    a_el = res.select_one(".result__url")
                    
                    if t_el:
                        title = t_el.get_text(strip=True)
                        snippet = s_el.get_text(strip=True) if s_el else ""
                        a_tag = t_el.select_one("a")
                        raw_href = a_tag["href"] if a_tag and "href" in a_tag.attrs else (a_el.get_text(strip=True) if a_el else "")
                        real_url = _decode_redirect_url(raw_href)
                        
                        if real_url and not _is_excluded_source(real_url):
                            hits.append(SourceHit(url=real_url, title=title, snippet=snippet, origin="web"))
                            if len(hits) >= max_results:
                                break
    except Exception as e:
        print(f"[discover] DDG search error: {e}")
    return hits


async def _bing_search(query: str, max_results: int) -> list[SourceHit]:
    """Bing Search HTML parsing."""
    url = "https://www.bing.com/search"
    hits = []
    try:
        async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
            resp = await client.get(url, params={"q": query}, headers=BROWSER_HEADERS)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for item in soup.select("li.b_algo"):
                    a_el = item.select_one("h2 a")
                    snippet_el = item.select_one(".b_caption p") or item.select_one(".b_algoSlug")
                    if a_el and a_el.get("href"):
                        title = a_el.get_text(strip=True)
                        raw_url = a_el.get("href", "")
                        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                        real_url = _decode_redirect_url(raw_url)
                        if real_url and not _is_excluded_source(real_url):
                            hits.append(SourceHit(url=real_url, title=title, snippet=snippet, origin="web"))
                            if len(hits) >= max_results:
                                break
    except Exception as e:
        print(f"[discover] Bing search error: {e}")
    return hits


async def _serpapi_search(query: str, max_results: int) -> list[SourceHit]:
    """SerpAPI fallback when key is provided."""
    if not SERPAPI_KEY:
        return []
    hits = []
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(
                SERPAPI_URL,
                params={"q": query, "api_key": SERPAPI_KEY, "num": max_results * 2}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("organic_results", []):
                    url = _clean_url(item.get("link", ""))
                    if url and not _is_excluded_source(url):
                        hits.append(SourceHit(
                            url=url,
                            title=item.get("title", ""),
                            snippet=item.get("snippet", ""),
                            origin="web"
                        ))
                        if len(hits) >= max_results:
                            break
    except Exception as e:
        print(f"[discover] SerpAPI search error: {e}")
    return hits


def _score_hit(hit: SourceHit, part_number: str, brand: str) -> float:
    """Ranks search hits so manufacturer datasheets, official PDFs, and spec pages rank highest."""
    score = 0.0
    url_lower = hit.url.lower()
    title_lower = (hit.title or "").lower()
    snip_lower = (hit.snippet or "").lower()
    
    pn_clean = "".join(c for c in part_number if c.isalnum()).lower()
    brand_clean = (brand or "").lower().strip()
    
    # Part number presence (highest value)
    if pn_clean and (pn_clean in url_lower.replace("-", "").replace("_", "") or pn_clean in title_lower.replace("-", "")):
        score += 25.0
    elif pn_clean and pn_clean in snip_lower.replace("-", ""):
        score += 15.0
        
    # Brand presence
    if brand_clean and len(brand_clean) >= 3:
        if brand_clean in url_lower or brand_clean in title_lower:
            score += 10.0
            
    # Technical document types
    if url_lower.endswith(".pdf") or "pdf" in url_lower or "datasheet" in url_lower:
        score += 20.0
    if any(kw in url_lower for kw in ["/product", "/spec", "catalog", "manual", "components"]):
        score += 10.0
        
    # Rich technical specs in snippet
    spec_signals = ["read", "write", "speed", "voltage", "capacity", "bearing", "dimensions", "mm", "v", "tbw", "iops", "nand", "sata", "m.2"]
    matched_signals = sum(1 for s in spec_signals if s in snip_lower)
    score += matched_signals * 3.0
    
    return score


async def discover_sources(product: ProductInput, max_results: int = 6) -> list[SourceHit]:
    """
    Multi-tier discovery engine:
      1. RAG Store: Ingested local datasets & catalog chunks.
      2. Multi-Engine Web Search: DuckDuckGo + Bing + SerpAPI concurrently.
      3. Intelligent Scoring & De-duplication: Prioritizes verified datasheets and specs.
    """
    rag_hits = []
    if rag_store.ready:
        query_text = f"{product.brand} {product.part_number} {product.short_description}".strip()
        rag_hits = rag_store.query(query_text, top_k=max_results, part_number=product.part_number)

    needed_web = max_results - len(rag_hits)
    if needed_web <= 0:
        return rag_hits[:max_results]

    brand_clean = (product.brand or "").strip()
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown"]
    if brand_clean.lower() in bad_brands:
        brand_clean = ""

    pn = product.part_number.strip()
    
    # Search query variants
    primary_query = f"{pn} {brand_clean} datasheet specifications".strip()
    secondary_query = f'"{pn}" {brand_clean}'.strip()
    
    all_hits = list(rag_hits)
    seen_urls = {h.url for h in all_hits if h.url}
    
    # 1. SerpAPI (if key present)
    if SERPAPI_KEY:
        serp_hits = await _serpapi_search(primary_query, max_results=needed_web)
        for h in serp_hits:
            if h.url not in seen_urls:
                seen_urls.add(h.url)
                all_hits.append(h)

    # 2. DuckDuckGo HTML Search
    if len(all_hits) < max_results:
        ddg_hits = await _ddg_search(primary_query, max_results=needed_web)
        for h in ddg_hits:
            if h.url not in seen_urls:
                seen_urls.add(h.url)
                all_hits.append(h)

    # 3. Bing Search
    if len(all_hits) < max_results:
        bing_hits = await _bing_search(secondary_query if secondary_query else primary_query, max_results=needed_web)
        for h in bing_hits:
            if h.url not in seen_urls:
                seen_urls.add(h.url)
                all_hits.append(h)

    # If still need hits, try broad query
    if len(all_hits) < max_results:
        fallback_query = f"{pn} {product.short_description}".strip()
        extra_ddg = await _ddg_search(fallback_query, max_results=needed_web)
        for h in extra_ddg:
            if h.url not in seen_urls:
                seen_urls.add(h.url)
                all_hits.append(h)

    # Rank all discovered hits
    all_hits.sort(key=lambda h: _score_hit(h, product.part_number, brand_clean), reverse=True)
    return all_hits[:max_results]
