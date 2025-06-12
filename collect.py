import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import database
from database import Database

WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 1
FINGERPRINTING_URL = "http://localhost:5000"
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def is_server_running(host='127.0.0.1', port=5000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def retrieve_traces_from_backend(driver):
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all Results')]")
    clear_button.click()
    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))

def is_collection_complete():
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

# ================== YOUR TASK 3 IMPLEMENTATION STARTS HERE ==================

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return the latest trace array, or None on error
    """
    try:
        # 1. Open the fingerprinting website
        driver.get(FINGERPRINTING_URL)
        time.sleep(1.5)

        # 2. Click the button to collect trace
        wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Collect Trace')]")))
        collect_btn = driver.find_element(By.XPATH, "//button[contains(., 'Collect Trace')]")

        # 3. Open the target website in a new tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(website_url)
        time.sleep(2.5)  # Wait for site to load

        # 4. Interact with the target website (scroll)
        try:
            body = driver.find_element(By.TAG_NAME, "body")
            for _ in range(2):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.8)
        except Exception:
            pass

        # 5. Return to the fingerprinting tab and close target website tab
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(0.8)

        collect_btn.click()  # Start trace collection

        # 6. Wait for the trace to be collected
        status_xpath = "//div[@role='alert']"
        wait.until(EC.text_to_be_present_in_element((By.XPATH, status_xpath), "Trace data collected"))

        # 7. Retrieve the latest trace from backend
        traces = retrieve_traces_from_backend(driver)
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

def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 25)
    total_new = 0
    for idx, website in enumerate(WEBSITES):
        current_collected = database.db.get_traces_collected().get(website, 0)
        num_to_collect = TRACES_PER_SITE if target_counts is None else target_counts.get(website, TRACES_PER_SITE)
        num_needed = num_to_collect - current_collected
        print(f"{website}: Need {num_needed} more traces.")

        for _ in range(num_needed):
            trace = collect_single_trace(driver, wait, website)
            if trace:
                database.db.save_trace(website, idx, trace)
                total_new += 1
                clear_trace_results(driver, wait)
                print(f"  - Saved trace for {website} (index {idx})")
            else:
                print(f"  - Trace collection failed for {website} (index {idx})")
    return total_new

# ================== YOUR TASK 3 IMPLEMENTATION ENDS HERE ====================

def main():
    if not is_server_running():
        print("Flask server is not running on localhost:5000. Start the backend before collecting.")
        return

    database.db.init_database()
    driver = None
    try:
        driver = setup_webdriver()
        while not is_collection_complete():
            new_traces = collect_fingerprints(driver)
            print(f"Collected {new_traces} new traces in this batch.")
            time.sleep(2)

        print("All traces collected!")
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()
        database.db.export_to_json(OUTPUT_PATH)

if __name__ == "__main__":
    main()
