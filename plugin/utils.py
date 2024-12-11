# utils.py
import re

def extract_urls(text: str):
    url_regex = r'(https?://\S+)'
    return re.findall(url_regex, text)

def generate_bypass_links(url: str):
    links = {
        "original": url,
        "12ft": f"https://12ft.io/{url}",
        "archive": f"https://web.archive.org/web/{url}",
    }
    return links

async def scrape_content(url: str):
    # For testing just output the url
    return url
    # Placeholder for actual scraping logic if needed in the future