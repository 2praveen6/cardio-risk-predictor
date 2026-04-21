import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        return webdriver.Chrome(options=options)
    except Exception:
        edge_options = webdriver.EdgeOptions()
        edge_options.add_argument('--headless')
        return webdriver.Edge(options=edge_options)

def clear_and_send(driver, aria_label, value):
    """Utility to clear a Streamlit number_input and send new values"""
    elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, f"//input[@aria-label='{aria_label}']"))
    )
    # Streamlit React bindings can be stubborn, so we send multiple backspaces
    elem.send_keys(Keys.CONTROL + "a")
    elem.send_keys(Keys.BACKSPACE)
    # Just in case for mac / standard length
    for _ in range(5):
        elem.send_keys(Keys.BACKSPACE)
    elem.send_keys(str(value))
    # Send TAB to trigger React's onBlur and register the change
    elem.send_keys(Keys.TAB)
    time.sleep(0.5)

def select_dropdown(driver, label_text, option_text):
    """Utility to select an item from a Streamlit selectbox"""
    # Find the nearest baseweb select container below the label
    dropdown = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, f"//label[descendant::*[contains(text(), '{label_text}')]]/following-sibling::div//div[@data-baseweb='select']"))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
    dropdown.click()
    time.sleep(0.5)
    
    # Click the listbox option
    option = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, f"//li[descendant::*[text()='{option_text}']]"))
    )
    option.click()
    time.sleep(0.5)

def run_test(driver, profile_name, expected_string, inputs):
    print(f"\n--- Testing Scenario: {profile_name} ---")
    driver.get("https://cardiorisknet.streamlit.app/")
    time.sleep(3)
    
    # Switch to Streamlit iframe
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    if iframes:
        driver.switch_to.frame(iframes[0])

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'CardioRisk AI')]"))
    )
    print("App loaded. Injecting values...")

    # Input values
    clear_and_send(driver, "Age (years)", inputs['age'])
    clear_and_send(driver, "Systolic BP (mmHg)", inputs['sys_bp'])
    clear_and_send(driver, "Diastolic BP (mmHg)", inputs['dia_bp'])
    clear_and_send(driver, "Heart Rate / Pulse (bpm)", inputs['heart_rate'])
    clear_and_send(driver, "BMI (kg/m²)", inputs['bmi'])
    clear_and_send(driver, "Total Cholesterol (mg/dL)", inputs['chol'])
    clear_and_send(driver, "HDL (mg/dL)", inputs['hdl'])
    clear_and_send(driver, "LDL (mg/dL)", inputs['ldl'])
    
    # Handle dropdown
    select_dropdown(driver, "Smoking Status", inputs['smoking'])

    print("Data injected. Clicking 'Predict Risk'...")
    predict_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[descendant::*[contains(text(), 'Predict Risk')]]"))
    )
    driver.execute_script("arguments[0].click();", predict_btn)

    print("Waiting for results...")
    result_elem = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.XPATH, f"//*[contains(text(), ' Risk') and (contains(text(), 'Low') or contains(text(), 'Moderate') or contains(text(), 'High'))]"))
    )
    
    # We grab all text in the results area to parse the % and the Risk Label
    body_text = driver.find_element(By.TAG_NAME, "body").text
    if expected_string in body_text:
        print(f"[OK] Success! The model logically analyzed the inputs and resulted in: {expected_string}")
        return True
    else:
        print(f"[FAIL] Uh oh. Expected {expected_string} but it wasn't found in the analysis.")
        return False

if __name__ == "__main__":
    driver = get_driver()
    try:
        # Scenario 1: Perfectly Healthy Young Adult -> Should be Low Risk
        healthy = {
            'age': 30, 'sys_bp': 110, 'dia_bp': 70, 'heart_rate': 60, 'bmi': 21.0,
            'chol': 140, 'hdl': 65, 'ldl': 80, 'smoking': 'Never'
        }
        run_test(driver, "Young & Healthy Profile", "Low Risk", healthy)

        # Scenario 2: Unhealthy Senior -> Should be High Risk
        high_risk = {
            'age': 82, 'sys_bp': 190, 'dia_bp': 110, 'heart_rate': 95, 'bmi': 38.0,
            'chol': 310, 'hdl': 25, 'ldl': 190, 'smoking': 'Current'
        }
        run_test(driver, "Elderly & High-Risk Profile", "High Risk", high_risk)
        
    finally:
        driver.quit()
