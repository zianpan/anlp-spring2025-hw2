import os
import requests
from bs4 import BeautifulSoup

def fix_url(href):
    href = href.strip()
    base_url = "https://events.cmu.edu/"
    if href.startswith("?"):
        href = base_url + href
    elif href.startswith("/") and not href.startswith("http"):
        href = base_url + href.lstrip("/")
    return href

def extract_event_details(soup):
    events = []
    seen = set()  # To track unique events based on (title, link)
    for event_div in soup.find_all("div", class_=lambda x: x and "lw_cal_event" in x):
        title = None
        link = None

        # Extract title and link from the container that holds them.
        title_container = event_div.find("div", class_="lw_events_title")
        if title_container:
            a_tag = title_container.find("a", href=True)
            if a_tag:
                title = a_tag.get_text(strip=True)
                link = fix_url(a_tag["href"])

        # Extract location, time, and summary if available.
        location_tag = event_div.find("div", class_="lw_events_location")
        location = location_tag.get_text(strip=True) if location_tag else ""
        
        time_tag = event_div.find("div", class_="lw_events_time")
        time_text = time_tag.get_text(strip=True) if time_tag else ""
        
        summary_tag = event_div.find("div", class_="lw_events_summary")
        summary = summary_tag.get_text(strip=True) if summary_tag else ""
        
        event_data = {
            "title": title,
            "link": link,
            "location": location,
            "time": time_text,
            "summary": summary,
        }
        
        # Only add the event if at least one of title or link is found and it's unique.
        if title or link:
            key = (title, link)
            if key not in seen:
                seen.add(key)
                events.append(event_data)
    return events

def fetch_detail_text(url):
    detail_text = ""
    try:
        print(f"Fetching details from: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            detail_soup = BeautifulSoup(response.text, "html.parser")
            detail_container = detail_soup.find("div", class_="events-internal")
            if detail_container:
                detail_text = detail_container.get_text(separator=" ", strip=True)
            else:
                detail_text = detail_soup.get_text(separator=" ", strip=True)
        else:
            detail_text = f"Error: Status code {response.status_code}"
    except Exception as e:
        detail_text = f"Error fetching details: {e}"
    return detail_text

def main():
    os.chdir("..")
    with open("cmu_event.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    events = extract_event_details(soup)
    print(f"Found {len(events)} unique events in the HTML file.")
    
    for event in events:
        if event["link"]:
            event["detail_text"] = fetch_detail_text(event["link"])
        else:
            event["detail_text"] = "N/A"
    
    output_events = "data/cmu_events.txt"
    os.makedirs(os.path.dirname(output_events), exist_ok=True)
    with open(output_events, "w", encoding="utf-8") as txt_file:
        for event in events:
            fields = [
                "event: " + (event["title"] or "N/A"),
                "location: " + (event["location"] or "N/A"),
                "time: " + (event["time"] or "N/A"),
                "summary: " + (event["summary"] or "N/A"),
                "detail: " + (event["detail_text"] or "N/A")
            ]
            line = "  ".join(fields)
            txt_file.write(line + "\n")
    
    # output_links = "data/cmu_links.txt"
    # with open(output_links, "w", encoding="utf-8") as link_file:
    #     for event in events:
    #         if event["link"]:
    #             link_file.write(event["link"] + "\n")
    
    print(f"Extraction complete. Data saved to {output_events}.")

if __name__ == "__main__":
    main()
