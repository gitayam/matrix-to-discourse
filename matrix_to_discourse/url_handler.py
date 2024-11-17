# url_handler.py

import re
import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def extract_urls(text: str) -> List[str]:
    url_regex = r'(https?://\S+)'
    return re.findall(url_regex, text)


def generate_bypass_links(url: str) -> Dict[str, str]:
    links = {
        "original": url,
        "12ft": f"https://12ft.io/{url}",
        "archive": f"https://web.archive.org/web/{url}",
    }
    return links


async def scrape_content(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve content from {url}: HTTP {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Remove scripts and styles
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                text = soup.get_text(separator='\n')
                return text.strip()
    except Exception as e:
        logger.error(f"Error scraping content from {url}: {e}")
        # Log or handle the exception as needed
        return None