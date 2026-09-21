"""
STAGE 2: Extract (Document & Web Content Intelligence)

Takes a discovered source (a URL) and pulls out usable raw text -- whether it's
a rich webpage, an HTML spec table, or a PDF datasheet.

If a remote server blocks scraping with 403 or times out, it gracefully falls back
to the verified search snippet and title so no search intelligence is lost.
"""
import httpx
import io
import re
import json
import logging
from bs4 import BeautifulSoup
import pdfplumber
from models import SourceHit

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

MAX_CHARS = 25000

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Upgrade-Insecure-Requests": "1"
}


async def extract_text(source: SourceHit) -> SourceHit:
    """
    Fetches the URL and extracts readable text, spec tables, and metadata.
    Mutates and returns the SourceHit with raw_text filled in.
    """
    if source.origin == "rag" and source.raw_text:
        return source

    if not source.url:
        source.raw_text = source.snippet or ""
        return source

    try:
        async with httpx.AsyncClient(
            timeout=8.0, follow_redirects=True, headers=BROWSER_HEADERS
        ) as client:
            resp = await client.get(source.url)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "").lower()
                if "pdf" in content_type or source.url.lower().endswith(".pdf"):
                    extracted_text = _extract_pdf_text(resp.content)
                else:
                    extracted_text = _extract_html_text(resp.text)
                
                if extracted_text and len(extracted_text.strip()) >= 50:
                    source.raw_text = extracted_text[:MAX_CHARS]
                    return source
    except Exception as e:
        pass

    # Fallback to search snippet and title
    fallback_parts = []
    if source.title:
        fallback_parts.append(f"Title: {source.title}")
    if source.snippet:
        fallback_parts.append(f"Snippet: {source.snippet}")
    source.raw_text = "\n".join(fallback_parts) if fallback_parts else None
    return source


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes with pdfplumber."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for i, page in enumerate(pdf.pages[:5]):  # inspect first 5 pages
                text = page.extract_text()
                if text:
                    pages_text.append(f"--- Page {i+1} ---\n{text}")
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        clean_row = [str(c).strip() for c in row if c is not None]
                        if len(clean_row) >= 2:
                            pages_text.append(" | ".join(clean_row))
            return "\n\n".join(pages_text)
    except Exception:
        return ""


def _extract_html_text(html: str) -> str:
    """Extracts title, meta descriptions, specification tables, definition lists, and lists."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        if tag.get("type") != "application/ld+json":
            tag.decompose()

    extracted_lines = []

    # 1. Page title
    if soup.title and soup.title.string:
        extracted_lines.append(f"Title: {soup.title.string.strip()}")

    # 2. Meta description & og tags
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        content = meta.get("content")
        if content and name in ["description", "keywords", "og:description", "og:title"]:
            extracted_lines.append(f"Meta {name}: {content.strip()}")

    # 3. JSON-LD structured product data
    for s in soup.find_all("script", type="application/ld+json"):
        if s.string:
            try:
                data = json.loads(s.string.strip())
                extracted_lines.append(f"JSON-LD Product: {json.dumps(data)}")
            except Exception:
                pass

    # 4. Specification Tables (<tr><td>...</td></tr>)
    for table in soup.find_all("table"):
        table_rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
            cols = [c for c in cols if c]
            if len(cols) >= 2:
                table_rows.append(" : ".join(cols[:4]))
        if table_rows:
            extracted_lines.append("\n[Spec Table]\n" + "\n".join(table_rows))

    # 5. Definition Lists (<dl><dt>...</dt><dd>...</dd></dl>)
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            k = dt.get_text(strip=True)
            v = dd.get_text(strip=True)
            if k and v:
                extracted_lines.append(f"{k}: {v}")

    # 6. Feature lists (<ul>, <li>)
    for ul in soup.find_all(["ul", "ol"]):
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        items = [it for it in items if len(it) > 5 and any(c.isalnum() for c in it)]
        if len(items) >= 2:
            extracted_lines.append("\n[Features]\n" + "\n".join(f"- {it}" for it in items[:15]))

    # 7. Remaining clean text
    body_text = soup.get_text(separator="\n", strip=True)
    body_clean = "\n".join(line.strip() for line in body_text.splitlines() if len(line.strip()) > 20)
    if body_clean:
        extracted_lines.append("\n[Body Content]\n" + body_clean[:8000])

    return "\n\n".join(extracted_lines)
