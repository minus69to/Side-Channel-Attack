import time
import json
import os
import signal
import sys
import platform
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys

# ... [rest of your file and helpers remain unchanged] ...

WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

FINGERPRINTING_URL = "http://localhost:5000"



def is_server_running(host='127.0.0.1', port=5000):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def collect_single_trace(driver, wait, website_url, website_index):
    """
    1. Open the fingerprinting website
    2. Open the target website in a new tab
    3. Interact with the target website (scroll, click, etc.)
    4. Return to the fingerprinting tab
    5. Click the button to collect trace
    6. Wait for the trace to be collected
    7. Retrieve and return the last collected trace
    """
    try:
        # 1. Open attacker page
        driver.get(FINGERPRINTING_URL)
        time.sleep(1.5)

        wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Collect Trace')]")))
        collect_btn = driver.find_element(By.XPATH, "//button[contains(., 'Collect Trace')]")

        # 2. Open target website in a new tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(website_url)
        time.sleep(2.5)  # Wait for site to load

        # 3. Interact (scroll)
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            for _ in range(2):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.8)
        except Exception:
            pass

        # 4. Return to attacker tab
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(0.8)

        # 5. Click "Collect Trace"
        collect_btn.click()

        # 6. Wait for completion
        status_xpath = "//div[@role='alert']"
        wait.until(EC.text_to_be_present_in_element((By.XPATH, status_xpath), "Trace data collected"))

        # 7. Get the latest trace from backend
        traces = driver.execute_script("""
            return fetch('/api/get_results')
                .then(r => r.ok ? r.json() : {traces: []})
                .then(data => data.traces || [])
                .catch(() => []);
        """)
        trace = traces[-1] if traces else None

        # Always close the victim tab
        if len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[1])
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        return trace
    except Exception as e:
        print(f"Error collecting trace: {e}")
        return None

def collect_fingerprints(driver):
    """
    For each website, collect one trace and return the dataset as a list of dicts.
    """
    from selenium.webdriver.support.ui import WebDriverWait
    wait = WebDriverWait(driver, 25)
    dataset = []
    for idx, website in enumerate(WEBSITES):
        print(f"Collecting for: {website} (index {idx})")
        trace = collect_single_trace(driver, wait, website, idx)
        if trace:
            dataset.append({
                "website": website,
                "website_index": idx,
                "trace_data": trace
            })
            print(f"  - Saved trace for {website}")
        else:
            print(f"  - Trace collection failed for {website}")
    return dataset

def main():
    """
    1. Check Flask server
    2. Set up Selenium
    3. Collect fingerprints
    4. Write dataset.json and metadata.json
    """
    if not is_server_running():
        print("Flask server is not running on localhost:5000. Start the backend before collecting.")
        return

    driver = None
    try:
        driver = setup_webdriver()
        dataset = collect_fingerprints(driver)
        # Write dataset.json
        with open("dataset.json", "w") as f:
            json.dump(dataset, f, indent=4)

        # Write metadata.json
        plat = platform.system().lower()
        if plat.startswith("win"):
            os_code = "w"
        elif plat.startswith("linux"):
            os_code = "l"
        elif plat.startswith("darwin") or "mac" in plat:
            os_code = "m"
        else:
            os_code = "u"  # unknown

        with open("metadata.json", "w") as f:
            json.dump({"os": os_code}, f, indent=4)

        print("Trace collection and export complete.")

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
