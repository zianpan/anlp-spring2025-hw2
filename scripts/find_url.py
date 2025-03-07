import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

def extract_event_links(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_divs = soup.find_all('div', class_='fdn-pres-content')
        
        event_urls = []
        for div in content_divs:
            # Find the link within the div
            link = div.find('a')
            if link and link.has_attr('href'):
                # Get the URL
                href = link['href']
                
                full_url = urljoin(url, href)
                
                event_urls.append(full_url)
        
        return event_urls
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []

def check_existing_urls(output_file):
    existing_urls = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                existing_urls.add(line.strip())
    
    return existing_urls

def save_to_txt(urls, existing_urls, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    new_urls = [url for url in urls if url not in existing_urls]
    
    if new_urls:
        with open(output_file, 'a', encoding='utf-8') as file:
            for url in new_urls:
                file.write(f"{url}\n")
    
    return len(new_urls)

def main():
    output_file = "output/urls.txt"
    
    for i in range(11, 23):
        url = f"https://www.pghcitypaper.com/pittsburgh/EventSearch?page={i}&sortType=date&v=d"

        print(f"Extracting event links from {url}...")
        urls = extract_event_links(url)
        
        if urls:
            print(f"Found {len(urls)} event URLs")
            
            # Check for existing URLs
            existing_urls = check_existing_urls(output_file)
            print(f"Found {len(existing_urls)} existing URLs in {output_file}")
            
            # Save new URLs to the text file
            added_count = save_to_txt(urls, existing_urls, output_file)
            
            print(f"Added {added_count} new URLs to {output_file}")
            
            # Show duplicate count
            duplicate_count = len(urls) - added_count
            if duplicate_count > 0:
                print(f"Skipped {duplicate_count} URLs that already exist in the file")
        else:
            print("No event URLs found or there was an error fetching the page.")

if __name__ == "__main__":
    main()