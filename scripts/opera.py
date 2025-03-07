import os
import requests
from bs4 import BeautifulSoup

os.chdir("..")
base_url = "https://www.pittsburghopera.org"
input_file = "opera4.html"
output_file = "data/opera_events4.txt"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

event_containers = soup.find_all("div", class_="event")

all_events = []

for event in event_containers:
    title = event.find("h4").get_text(strip=True) if event.find("h4") else "N/A"
    p_tags = event.find_all("p")
    date_text = p_tags[0].get_text(strip=True) if len(p_tags) > 0 else "N/A"
    time_text = p_tags[1].get_text(strip=True) if len(p_tags) > 1 else "N/A"
    location_text = p_tags[2].get_text(strip=True) if len(p_tags) > 2 else "N/A"
    
    a_tag = event.find(
        "a",
        href=lambda x: x and (
            x.startswith("/calendar/") or
            x.startswith("https://pittsburghopera.org/season/") or
            x.startswith("https://www.pittsburghopera.org/season/")
        )
    )
    if not a_tag:
        # If no valid link, skip this event.
        continue

    detail_rel_link = a_tag["href"].strip()
    # If the link is relative, prefix with the base URL.
    if detail_rel_link.startswith("/"):
        detail_full_link = base_url + detail_rel_link
    else:
        detail_full_link = detail_rel_link


    detail_text = ""
    try:
        print(f"Fetching details from: {detail_full_link}")
        response = requests.get(detail_full_link)
        if response.status_code == 200:
            detail_soup = BeautifulSoup(response.text, "html.parser")
            # Try to get detail text from a container with class "events-internal"
            detail_container = detail_soup.find("div", class_="events-internal")
            if detail_container:
                detail_text = detail_container.get_text(separator=" ", strip=True)
            else:
                detail_text = detail_soup.get_text(separator=" ", strip=True)
        else:
            detail_text = f"Error: Status code {response.status_code}"
    except Exception as e:
        detail_text = f"Error fetching details: {e}"

    event_data = {
        "title": title,
        "date": date_text,
        "time": time_text,
        "location": location_text,
        "detail_text": detail_text,
    }
    all_events.append(event_data)

with open(output_file, "w", encoding="utf-8") as f:
    for ev in all_events:
        line = (
            f"title: {ev['title']}  date: {ev['date']}  time: {ev['time']}  location: {ev['location']}  detail: {ev['detail_text']}"
        )
        f.write(line + "\n")

print(f"Extracted full details for {len(all_events)} events and saved to {output_file}")
