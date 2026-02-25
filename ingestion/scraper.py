"""Web scraper for AT&T public policy, legal, and support pages."""

import time
import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document



ATT_URLS = [
    {
        "url": "https://www.att.com/legal/terms.aup.html",
        "title": "AT&T Acceptable Use Policy",
    },
    {
        "url": "https://www.att.com/legal/privacy-policy.html",
        "title": "AT&T Privacy Policy",
    },
    {
        "url": "https://www.att.com/legal/terms.internetTermsOfService.html",
        "title": "AT&T Internet Terms of Service",
    },
    {
        "url": "https://www.att.com/support/article/wireless/KM1387497/",
        "title": "AT&T Wireless Network Troubleshooting",
    },
    {
        "url": "https://www.att.com/support/article/wireless/KM1008767/",
        "title": "AT&T Report Service Outage",
    },
    {
        "url": "https://www.att.com/legal/terms.broadbandNotice.html",
        "title": "AT&T Broadband Information",
    },
    {
        "url": "https://about.att.com/csr/home/governance/transparency.html",
        "title": "AT&T Transparency Report",
    },
]

# ── HTTP settings ────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_TIMEOUT = 15
POLITE_DELAY = 2


def _clean_text(raw_html: str) -> str:
    """Extract meaningful text from HTML, removing non-content elements."""
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                              "noscript", "iframe", "svg", "button", "form"]):
        tag.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|body|main|article", re.I))
        or soup.find("div", id=re.compile(r"content|body|main|article", re.I))
        or soup.body
        or soup
    )

    text = main_content.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def scrape_single_page(url: str, title: str) -> Optional[Document]:
    """Scrape a single URL and return a LangChain Document, or None on failure."""
    try:
        print(f"  🌐 Scraping: {title}")
        print(f"     URL: {url}")

        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        text = _clean_text(response.text)

        if len(text) < 200:
            print(f"     ⚠️  Skipped (too little content: {len(text)} chars)")
            return None

        doc = Document(
            page_content=text,
            metadata={
                "source": title,
                "url": url,
                "type": "web_scrape",
                "content_length": len(text),
            },
        )
        print(f"     ✅ Scraped {len(text)} characters")
        return doc

    except requests.exceptions.ConnectionError:
        print(f"     ❌ Connection failed (no internet or site unreachable)")
        return None
    except requests.exceptions.Timeout:
        print(f"     ❌ Request timed out after {REQUEST_TIMEOUT}s")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"     ❌ HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"     ❌ Unexpected error: {e}")
        return None


def scrape_att_pages(urls: Optional[List[dict]] = None) -> List[Document]:
    """Scrape multiple AT&T pages and return a list of Documents."""
    if urls is None:
        urls = ATT_URLS

    print(f"\n🌐 Scraping {len(urls)} AT&T page(s) …")
    documents: List[Document] = []

    for i, page_info in enumerate(urls):
        doc = scrape_single_page(page_info["url"], page_info["title"])
        if doc:
            documents.append(doc)

        if i < len(urls) - 1:
            time.sleep(POLITE_DELAY)

    print(f"\n✅ Successfully scraped {len(documents)}/{len(urls)} pages")
    return documents


def scrape_custom_url(url: str, title: Optional[str] = None) -> Optional[Document]:
    """Scrape a single user-provided URL for the Streamlit UI."""
    if title is None:
        title = f"Custom: {url[:60]}"
    return scrape_single_page(url, title)

