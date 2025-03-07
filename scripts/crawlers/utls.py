
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_all_urls(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"})
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all text from the page
        # page_text = soup.get_text(separator="\n", strip=True)

        # Extract all hyperlinks
        links = set()
        for link in soup.find_all("a", href=True):
            absolute_link = urljoin(url, link["href"])  # Convert relative links to absolute
            links.add(absolute_link)

        return links

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return None


def extract_clean_text(url):
    # Fetch webpage
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript", "iframe", "footer", "header", "nav", "aside"]):
        tag.extract()

    # Extract text from relevant tags
    text_elements = soup.find_all(['p', 'div', 'span', 'article', 'li'])

    # Extract and clean text
    paragraphs = [elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)]

    # Join paragraphs with double newlines for readability
    cleaned_text = "\n\n".join(paragraphs)

    return cleaned_text
