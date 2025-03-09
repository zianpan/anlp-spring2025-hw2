import time
import csv
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def random_sleep(min_seconds=1, max_seconds=3):
    time.sleep(random.uniform(min_seconds, max_seconds))

url = "https://www.nhl.com/penguins/schedule/2025-03-01/list"

chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    });
    """
})

driver.get(url)
print("Page loaded. Waiting for content...")
random_sleep(3, 6)

all_games = []
current_page = 1
max_pages = 12

def extract_gamecenter_links():
    gamecenter_links = []
    
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.rt-table, div[role='table']"))
        )
        random_sleep(1, 2)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        links = soup.select("a[href*='/gamecenter/']")
        print(f"Found {len(links)} gamecenter links on this page")
        
        for link in links:
            href = link.get('href')
            if href:
                if href.startswith('/'):
                    href = f"https://www.nhl.com{href}"
                
                game_info = {
                    'link': href,
                    'text': link.get_text(strip=True)
                }
                
                parent_row = link.find_parent('tr') or link.find_parent('div[role="row"]')
                if parent_row:
                    cells = parent_row.select("td, th, div.rt-td, div[role='cell']")
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        if cell_text:
                            game_info[f'cell_{i}'] = cell_text
                
                gamecenter_links.append(game_info)
    
    except Exception as e:
        print(f"Error extracting gamecenter links: {str(e)}")
    
    return gamecenter_links

def extract_gamecenter_details(game_info):
    try:
        print(f"Navigating to gamecenter: {game_info['link']}")
        driver.get(game_info['link'])
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        random_sleep(2, 4)
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        team_elements = soup.select(".team-name, .team-logo-container h2, .team-name-container")
        if team_elements and len(team_elements) >= 2:
            game_info['home_team'] = team_elements[0].get_text(strip=True) 
            game_info['away_team'] = team_elements[1].get_text(strip=True)
        
        score_elements = soup.select(".score-value, .score-container .score")
        if score_elements and len(score_elements) >= 2:
            game_info['home_score'] = score_elements[0].get_text(strip=True)
            game_info['away_score'] = score_elements[1].get_text(strip=True)
        
        date_time_element = soup.select_one(".game-date, .date-time-container")
        if date_time_element:
            game_info['date_time'] = date_time_element.get_text(strip=True)
        
        status_element = soup.select_one(".game-status, .status-container")
        if status_element:
            game_info['status'] = status_element.get_text(strip=True)
        
        venue_element = soup.select_one(".venue-info, .venue-container")
        if venue_element:
            game_info['venue'] = venue_element.get_text(strip=True)
        
        if 'home_team' not in game_info or 'away_team' not in game_info:
            team_containers = soup.select(".team-container, .team-info, .team-banner")
            for i, container in enumerate(team_containers[:2]):
                team_text = container.get_text(strip=True)
                if i == 0:
                    game_info['home_team_alt'] = team_text
                else:
                    game_info['away_team_alt'] = team_text
        
        data_sections = soup.select(".stats-container, .game-stats, .highlights-container")
        for i, section in enumerate(data_sections):
            section_text = section.get_text(strip=True, separator=' | ')
            game_info[f'additional_data_{i}'] = section_text
            
        shots_elements = soup.select(".shots-container, .shots-value")
        if shots_elements and len(shots_elements) >= 2:
            game_info['home_shots'] = shots_elements[0].get_text(strip=True)
            game_info['away_shots'] = shots_elements[1].get_text(strip=True)
        
        print(f"Successfully extracted details for game: {game_info.get('home_team', '')} vs {game_info.get('away_team', '')}")
            
    except Exception as e:
        print(f"Error extracting gamecenter details: {str(e)}")
    
    return game_info

def try_next_button():
    try:
        next_button_xpaths = [
            "//button[.//span[contains(text(),'Next')] and .//svg[@data-testid='ChevronRightIcon']]",
            "//button[.//span[contains(text(),'Next')]]",
            "//button[contains(@class, 'next') or contains(@class, 'Next')]",
            "//div[contains(@class, 'pagination')]//button[position()=last()]",
            "//button[.//svg[contains(@data-testid, 'ChevronRight')]]",
            "//button[@aria-label='Next Page']",
            "//button[contains(@class, 'hyDqtC')]"
        ]
        
        next_button_css = [
            "button.sc-ldgOGP.sc-faxByu",
            "button.hyDqtC",
            "div.pagination button:last-of-type",
            "button[aria-label='Next Page']",
            "button:has(svg[data-testid='ChevronRightIcon'])"
        ]
        
        for xpath in next_button_xpaths:
            try:
                elements = driver.find_elements(By.XPATH, xpath)
                if elements:
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            print(f"Found Next button using XPath: {xpath}")
                            return element
            except:
                continue
        
        for css in next_button_css:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, css)
                if elements:
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            print(f"Found Next button using CSS: {css}")
                            return element
            except:
                continue
                
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            try:
                html = button.get_attribute("outerHTML")
                if "Next" in html or "ChevronRight" in html or "→" in html:
                    if button.is_displayed() and button.is_enabled():
                        print("Found Next button by searching button HTML")
                        return button
            except:
                continue
                
        return None
    except Exception as e:
        print(f"Error finding next button: {str(e)}")
        return None

def click_next_button(button):
    if not button:
        return False
        
    attempts = 0
    max_attempts = 3
    success = False
    
    try:
        current_indicators = []
        indicator_elements = driver.find_elements(By.CSS_SELECTOR, ".pagination span, .pagination button, .rt-pagination-page")
        for elem in indicator_elements:
            if "active" in elem.get_attribute("class"):
                current_indicators.append(elem.text)
        print(f"Current page indicators: {current_indicators}")
    except:
        pass
    
    while attempts < max_attempts and not success:
        attempts += 1
        try:
            try:
                overlays = driver.find_elements(By.CSS_SELECTOR, ".modal, .overlay, .popup")
                for overlay in overlays:
                    if overlay.is_displayed():
                        close_buttons = overlay.find_elements(By.CSS_SELECTOR, "button.close, .close-button, button[aria-label='Close']")
                        for close in close_buttons:
                            if close.is_displayed():
                                close.click()
                                random_sleep(1, 2)
            except:
                pass
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            random_sleep(0.5, 1)
            
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            random_sleep(1, 2)
            
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, f"//{button.tag_name}[@class='{button.get_attribute('class')}']")))
            button.click()
            success = True
            print("Standard click successful")
        except Exception as e:
            print(f"Standard click failed: {str(e)}")
            try:
                driver.execute_script("arguments[0].click();", button)
                success = True
                print("JavaScript click successful")
            except Exception as e:
                print(f"JavaScript click failed: {str(e)}")
                try:
                    from selenium.webdriver.common.action_chains import ActionChains
                    
                    actions = ActionChains(driver)
                    actions.move_by_offset(-1000, -1000).perform()
                    random_sleep(0.5, 1)
                    
                    actions = ActionChains(driver)
                    actions.move_to_element(button).pause(1).click().perform()
                    success = True
                    print("ActionChains click successful")
                except Exception as e:
                    print(f"ActionChains click failed: {str(e)}")
                    
                    try:
                        from selenium.webdriver.common.keys import Keys
                        button.send_keys(Keys.ENTER)
                        success = True
                        print("Keyboard navigation click successful")
                    except Exception as e:
                        print(f"Keyboard click attempt {attempts} failed: {str(e)}")
                        random_sleep(1, 2)
    
    if success:
        print("Click appears successful. Waiting for page to update...")
        random_sleep(3, 5)
        return True
    return False

def has_page_changed(before_content, max_tries=3):
    tries = 0
    while tries < max_tries:
        tries += 1
        current_content = driver.page_source
        
        if len(current_content) != len(before_content):
            print("Page content length changed - confirmed page change")
            return True
            
        try:
            pagination_elements = driver.find_elements(By.CSS_SELECTOR, ".rt-pagination-page, .pagination-page, [aria-label*='page']")
            for elem in pagination_elements:
                if "active" in elem.get_attribute("class") and "2" in elem.text:
                    if current_page == 1:
                        print("Found active page 2 indicator - confirmed page change")
                        return True
                if "active" in elem.get_attribute("class") and str(current_page + 1) in elem.text:
                    print(f"Found active page {current_page + 1} indicator - confirmed page change")
                    return True
        except Exception as e:
            print(f"Error checking pagination indicators: {str(e)}")
        
        try:
            current_dates = []
            date_elements = driver.find_elements(By.CSS_SELECTOR, "span.date, div.date, td.date")
            for date_elem in date_elements:
                current_dates.append(date_elem.text)
            
            if current_dates and current_dates != getattr(driver, 'current_dates', []):
                print("Event dates changed - confirmed page change")
                driver.current_dates = current_dates
                return True
            if not hasattr(driver, 'current_dates'):
                driver.current_dates = current_dates
        except Exception as e:
            print(f"Error checking event dates: {str(e)}")
        
        print(f"Page content appears unchanged. Waiting... (try {tries}/{max_tries})")
        random_sleep(2, 3)
    
    return False

def use_url_pagination():
    current_url = driver.current_url
    print(f"Current URL: {current_url}")
    
    if "page=" in current_url:
        parts = current_url.split("page=")
        page_num = int(parts[1].split("&")[0])
        next_url = current_url.replace(f"page={page_num}", f"page={page_num+1}")
    else:
        if "?" in current_url:
            next_url = f"{current_url}&page={current_page+1}"
        else:
            next_url = f"{current_url}?page={current_page+1}"
    
    print(f"Attempting URL pagination to: {next_url}")
    
    before_content = driver.page_source
    
    driver.get(next_url)
    random_sleep(3, 5)
    
    current_content = driver.page_source
    if current_content != before_content:
        print("URL pagination successful - content changed")
        return True
    else:
        print("URL pagination failed - content unchanged")
        driver.get(current_url)
        random_sleep(2, 3)
        return False

try:
    seen_event_texts = set()
    consecutive_duplicate_pages = 0
    max_duplicate_pages = 2
    
    while current_page <= max_pages:
        print(f"\n==== Processing page {current_page} ====")
        
        before_content = driver.page_source
        
        page_links = extract_gamecenter_links()
        print(f"Found {len(page_links)} gamecenter links on page {current_page}")
        
        for game_info in page_links:
            current_page_url = driver.current_url
            
            detailed_game_info = extract_gamecenter_details(game_info)
            all_games.append(detailed_game_info)
            
            driver.get(current_page_url)
            random_sleep(2, 3)
        
        page_texts = set(str(link) for link in page_links)
        if page_texts.issubset(seen_event_texts) and page_links:
            consecutive_duplicate_pages += 1
            print(f"Warning: All links on this page are duplicates (consecutive duplicates: {consecutive_duplicate_pages})")
            if consecutive_duplicate_pages >= max_duplicate_pages:
                print(f"Stopping after {max_duplicate_pages} consecutive duplicate pages")
                break
        else:
            consecutive_duplicate_pages = 0
            seen_event_texts.update(page_texts)
        
        next_button = try_next_button()
        
        if next_button:
            print("Next button found. Attempting to click...")
            
            if click_next_button(next_button):
                if has_page_changed(before_content):
                    print("Successfully navigated to next page via button click")
                    current_page += 1
                    continue
                else:
                    print("Button click didn't change page. Trying alternative...")
            else:
                print("Could not click the Next button. Trying alternative...")
        else:
            print("No Next button found. Trying alternative pagination method...")
        
        if use_url_pagination():
            print("Successfully navigated to next page via URL pagination")
            current_page += 1
            continue
        
        try:
            month_selectors = driver.find_elements(By.CSS_SELECTOR, "button.month-selector, button[data-month], div.schedule-selector button")
            
            month_clicked = False
            for i, month_button in enumerate(month_selectors):
                if i == current_page:
                    print(f"Trying month selection pagination: {month_button.text}")
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", month_button)
                    random_sleep(1, 2)
                    
                    try:
                        month_button.click()
                        random_sleep(2, 4)
                        
                        if driver.page_source != before_content:
                            print("Successfully navigated via month selection")
                            current_page += 1
                            month_clicked = True
                            break
                    except:
                        print("Failed to click month selector")
                        continue
            
            if month_clicked:
                continue
                
            print("All pagination methods failed. Ending scrape.")
            break
                
        except Exception as e:
            print(f"Error trying month selection: {str(e)}")
            print("All pagination methods failed. Ending scrape.")
            break
            
except Exception as e:
    print(f"Error during scraping: {str(e)}")
finally:
    print(f"\nTotal games collected: {len(all_games)}")
    
    with open("game_details.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = set()
        for game in all_games:
            fieldnames.update(game.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        for game in all_games:
            csv_writer.writerow(game)
    
    with open("game_summaries.txt", "w", encoding="utf-8") as f:
        for i, game in enumerate(all_games):
            f.write(f"Game #{i+1}:\n")
            f.write(f"Link: {game.get('link', 'N/A')}\n")
            
            important_fields = ['home_team', 'away_team', 'home_score', 'away_score', 
                                'date_time', 'status', 'venue']
            
            for field in important_fields:
                if field in game:
                    f.write(f"{field.replace('_', ' ').title()}: {game[field]}\n")
            
            f.write("\n" + "-"*50 + "\n\n")

    driver.quit()
    print("Scraping completed successfully! Results saved to game_details.csv and game_summaries.txt")