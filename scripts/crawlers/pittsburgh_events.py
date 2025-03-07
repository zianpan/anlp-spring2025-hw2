from bs4 import BeautifulSoup

# Open and parse the HTML file.
import os
os.chdir("..")
with open("pittsburgh_events.html", "r", encoding="utf-8") as file:
    html = file.read()
soup = BeautifulSoup(html, "html.parser")

event_rows = soup.find_all("li", class_="date-row")

events = []
seen = set()

for row in event_rows:
    # Extract date details.
    date_div = row.find("div", class_="date")
    if date_div:
        month = date_div.find("div", class_="month").get_text(strip=True) if date_div.find("div", class_="month") else ""
        day = date_div.find("div", class_="day").get_text(strip=True) if date_div.find("div", class_="day") else ""
        year = date_div.find("div", class_="year").get_text(strip=True) if date_div.find("div", class_="year") else ""
    else:
        month = day = year = ""
    
    # Force month to uppercase (e.g., "MAR")
    month_upper = month.upper()
    
    # Build a full date string.
    date_str = f"{month_upper} {day} {year}"
    
    # Extract time and weekday.
    time_div = row.find("div", class_="time")
    time_text = ""
    week = ""
    if time_div:
        # Extract the text directly under the time div (excluding any nested tags)
        time_text = time_div.find(text=True, recursive=False)
        if time_text:
            time_text = time_text.strip()
        week_tag = time_div.find("div", class_="week")
        week = week_tag.get_text(strip=True) if week_tag else ""
    
    # Extract venue.
    venue_div = row.find("div", class_="venue")
    venue_text = venue_div.get_text(" ", strip=True) if venue_div else ""
    
    # Build a key for uniqueness.
    key = (month_upper, day, year, time_text, week, venue_text)
    if key in seen:
        continue
    seen.add(key)
    
    event = {
        "date": date_str,
        "month": month_upper,
        "day": day,
        "year": year,
        "time": time_text if time_text else "N/A",
        "week": week if week else "N/A",
        "venue": venue_text if venue_text else "N/A"
    }
    events.append(event)

# Save each unique event on one line with fields separated by two spaces.
with open("data/pittsburgh_events.txt", "w", encoding="utf-8") as outfile:
    for event in events:
        line = (
            f"date:{event['date']}  "
            f"month:{event['month']}  "
            f"day:{event['day']}  "
            f"year:{event['year']}  "
            f"time:{event['time']}  "
            f"week:{event['week']}  "
            f"venue:{event['venue']}"
        )
        outfile.write(line + "\n")

print(f"Extracted {len(events)} unique events and saved to data/pittsburgh_events.txt")
