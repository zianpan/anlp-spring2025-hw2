from bs4 import BeautifulSoup

def fix_url(href):
    """
    Clean up the URL.
    If the URL is relative (e.g. "bajaj.html"), prepend a base URL.
    Adjust the base_url value as needed.
    """
    href = href.strip()
    base_url = "https://www.cmu.edu/engage/alumni/get-involved/tartansontherise/2022/"
    # If href starts with a slash or doesn't start with http, prepend the base URL.
    if not href.startswith("http"):
        # Remove any leading slashes and then prepend the base URL.
        href = base_url + href.lstrip("/")
    return href

import os
os.chdir("..")
with open("tartan2023.html", "r", encoding="utf-8") as file:
    html = file.read()

soup = BeautifulSoup(html, "html.parser")

# Find all divs with class "photo"
photo_divs = soup.find_all("div", class_="photo")

unique_links = set()

for div in photo_divs:
    a_tag = div.find("a", href=True)
    if a_tag:
        url = fix_url(a_tag["href"])
        unique_links.add(url)

# Save each unique URL on one line in the output file.
with open("websites/tartan2022_urls.txt", "w", encoding="utf-8") as outfile:
    for url in unique_links:
        outfile.write(url + "\n")

print(f"Extracted {len(unique_links)} unique links and saved to websites/tartan2022_urls.txt")
