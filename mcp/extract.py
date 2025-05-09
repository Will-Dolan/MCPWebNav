"""
FUNCTIONS: extract_info_from_links(urls)
TAKES LIST OF URLS AS INPUT, OUTPUTS LIST OF DOCUMENTS CONTAINING KEY INFORMATION
the sleep time and max_tokens are parameters that can be changed to vary output length
"""
import requests
from bs4 import BeautifulSoup
import anthropic
import time

import os
from dotenv import load_dotenv

load_dotenv()

# Read API key from env variable
API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize Claude API client
client = anthropic.Anthropic(api_key=API_KEY)

# Function to fetch readable text from a webpage
def fetch_page_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return " ".join([p.text for p in soup.find_all("p")])  # Extract paragraphs
    else:
        return None

# Function to extract key info from a list of web sources
def extract_info_from_links(urls):
    summaries = []

    for i, url in enumerate(urls):
        print(f"Processing URL {i + 1}/{len(urls)}: {url}")  # Status update

        page_text = fetch_page_content(url)  # Fetch webpage text
        if page_text:
            print("  - Page content fetched.")  # Status update

            prompt = f"Extract key information from the following article content and respond in a plain text paragraph with no formatting. Do not preface the text in any way.\n\n{page_text}"

            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            summary_text = response.content[0].text
            summaries.append(summary_text)
            print("  - Summary extracted.")  # Status update

        else:
            print("  - Failed to fetch page content.")  # Status update

        time.sleep(5)  # Avoid rate limits

    print("All URLs processed.")  # Final status update
    return summaries

