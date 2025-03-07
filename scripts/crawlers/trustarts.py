import os
import requests
from bs4 import BeautifulSoup

os.chdir("..")
base_url = "https://trustarts.org"
input_file = "trustarts6.html"
output_file = "data/trustarts_events6.txt"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/112.0.0.0 Safari/537.36"
}

with open(input_file, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

event_articles = soup.find_all("article", class_="event")

all_events = []

for event in event_articles:
    title_tag = event.find("h3", class_="title")
    title = title_tag.get_text(strip=True) if title_tag else "N/A"
    
    a_tag = event.find("a", class_="lead-image-link")
    if a_tag and "href" in a_tag.attrs:
        detail_rel_link = a_tag["href"].strip()
        if detail_rel_link.startswith("/"):
            detail_link = base_url + detail_rel_link
        else:
            detail_link = detail_rel_link
    else:
        detail_link = "N/A"
    
    time_wrapper = event.find("div", class_="time-wrapper")
    time_text = time_wrapper.get_text(strip=True) if time_wrapper else "N/A"
    
    venue_tag = event.find("div", class_="venue")
    venue = venue_tag.get_text(strip=True) if venue_tag else "N/A"
    
    org_tag = event.find("div", class_="organization")
    organization = org_tag.get_text(strip=True) if org_tag else "N/A"
    
    cat_ul = event.find("ul", class_="category-group")
    if cat_ul:
        categories = [li.get_text(strip=True) for li in cat_ul.find_all("li", class_="category")]
        categories_text = ", ".join(categories)
    else:
        categories_text = "N/A"
    
    detail_text = ""
    if detail_link != "N/A":
        try:
            print(f"Fetching details from: {detail_link}")
            response = requests.get(detail_link, headers=headers)
            if response.status_code == 200:
                detail_soup = BeautifulSoup(response.text, "html.parser")
                container = detail_soup.find("div", class_="production-detail")
                if container:
                    detail_text = container.get_text(separator=" ", strip=True)
                else:
                    detail_text = detail_soup.get_text(separator=" ", strip=True)
            else:
                detail_text = f"Error: Status code {response.status_code}"
        except Exception as e:
            detail_text = f"Error fetching details: {e}"
    else:
        detail_text = "N/A"
    
    event_data = {
        "title": title,
        "time": time_text,
        "venue": venue,
        "organization": organization,
        "detail_text": detail_text,
    }
    
    all_events.append(event_data)

with open(output_file, "w", encoding="utf-8") as f:
    for ev in all_events:
        line = (
            f"title: {ev['title']}  "
            f"time: {ev['time']}  "
            f"venue: {ev['venue']}  "
            f"organization: {ev['organization']}  "
            f"detail: {ev['detail_text']}"
        )
        f.write(line + "\n")

print(f"Extracted details for {len(all_events)} events.")
print(f"Event data saved to {output_file}")