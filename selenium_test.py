import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_streamlit_app():
    print("Initializing Selenium WebDriver...")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print("Chrome WebDriver not found, falling back to Edge...")
        edge_options = webdriver.EdgeOptions()
        edge_options.add_argument('--headless')
        driver = webdriver.Edge(options=edge_options)

    url = "https://cardiorisknet.streamlit.app/"
    print(f"Navigating to {url} ...")
    driver.get(url)
    
    try:
        # 1. Wait for basic application structure to load
        print("Waiting for Streamlit app to 'wake up' and load...")
        
        # Streamlit Cloud wraps the app in an iframe
        time.sleep(5) # Let basic layout load
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            print(f"Found {len(iframes)} iframe(s). Switching to the first one...")
            driver.switch_to.frame(iframes[0])

        WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'CardioRisk AI')]"))
        )
        print("[OK] Website fully loaded. Title verified.")

        # 2. Click the Predict Risk button
        # Streamlit buttons have the inner struct `<button><div><p>Predict Risk</p></div></button>` 
        print("Locating 'Predict Risk' button...")
        predict_btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//button[descendant::*[contains(text(), 'Predict Risk')]]"))
        )
        
        print("[OK] Button found. Simulating user click...")
        
        # It's better to use javascript click for Streamlit elements if they are overlapped by custom CSS
        driver.execute_script("arguments[0].click();", predict_btn)

        # 3. Wait for the prediction output metric to populate in the results column
        print("Waiting for backend prediction models to compute and render...")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Low Risk') or contains(text(), 'Moderate Risk') or contains(text(), 'High Risk')]"))
        )
        print("[OK] Test Passed! Prediction generated correctly.")

    except Exception as e:
        print(f"[FAIL] Test Failed. Timeout or element not found: {e}")
        # Capture screenshot for debugging
        screenshot_path = "error_screenshot.png"
        driver.save_screenshot(screenshot_path)
        print(f"Saved layout screenshot to {screenshot_path}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_streamlit_app()
