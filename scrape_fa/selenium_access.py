from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import os

def enter_password(driver, password):
    """
    Find the first input or textarea element and enter the password.
    """
    input_element = None
    try:
        input_element = driver.find_element(By.TAG_NAME, 'input')
    except:
        raise Exception("No input element found on the page.")

    if input_element:
        input_element.click()
        input_element.send_keys(password)
        return True
    return False

def click_submit_button(driver):
    """
    Find and click the element with class 'submit'.
    """
    try:
        submit_element = driver.find_element(By.CLASS_NAME, 'submit')
        submit_element.click()
        print("Submit button clicked successfully.")
        return True
    except:
        print("No element with class 'submit' found.")
        return False

def click_element_by_text(driver, text):
    """
    Find and click an element that contains the specified text content.
    """
    try:
        # Use XPath to find element containing the specified text
        element = driver.find_element(By.XPATH, f"//*[contains(text(), '{text}')]")
        element.click()
        print(f"Element containing '{text}' clicked successfully.")
        return True
    except:
        print(f"No element containing '{text}' text found.")
        return False

def click_geocoder_input(driver):
    """
    Find and click the element with class 'mapboxgl-ctrl-geocoder--input'.
    """
    try:
        geocoder_input = driver.find_element(By.CLASS_NAME, 'mapboxgl-ctrl-geocoder--input')
        geocoder_input.click()
        print("Geocoder input field clicked successfully.")
        return True
    except:
        print("No element with class 'mapboxgl-ctrl-geocoder--input' found.")
        return False

def clear_geocoder_input(geocoder_input):
    """
    Clear the geocoder input field.
    """
    geocoder_input.click()  # Ensure the field is focused
    geocoder_input.send_keys(Keys.CONTROL + 'a')  # Select all text in the input field
    geocoder_input.send_keys(Keys.DELETE)  # Clear the input field
    print("Geocoder input field cleared successfully.")


def zoom_click(driver, zoom_clicks=10):
    """
    Double click on the mapboxgl-canvas element a fixed number of times (10).
    """
    try:
        canvas = driver.find_element(By.CLASS_NAME, 'mapboxgl-canvas')

        for attempt in range(zoom_clicks):
            print(f"Zoom click {attempt + 1}/{zoom_clicks}")

            # Double click on the canvas to zoom in
            ActionChains(driver).move_to_element(canvas).double_click().perform()
            time.sleep(0.5)  # Small delay between double clicks

        print(f"Completed {zoom_clicks} zoom clicks")
        return True
    except Exception as e:
        print(f"Error while double clicking to zoom: {e}")
        return False

def enter_coordinates_in_geocoder(driver, lat, lon):
    """
    Enter coordinates in the format "Lat: {} Lng: {}" into the geocoder input field.
    """
    geocoder_input = driver.find_element(By.CLASS_NAME, 'mapboxgl-ctrl-geocoder--input')
    geocoder_input.click()  # Ensure the field is focused
    coordinate_string = f"Lat: {lon} Lng: {lat}"  # DELIBERATELY swapped order to match the expected format
    geocoder_input.send_keys(coordinate_string)  # Enter the coordinates
    time.sleep(1)
    # Find and click the first suggestion title
    suggestion_title = driver.find_element(By.CLASS_NAME, 'mapboxgl-ctrl-geocoder--suggestion-title')
    suggestion_title.click()
    print(f"Entered coordinates: {coordinate_string} and clicked suggestion")

def take_screenshot(driver, i, j, tag):
    """
    Take a screenshot with coordinates in the filename.
    """
    try:
        filename = f"../{tag}s/screenshot_i_{i}_j_{j}_{tag}.png"
        driver.save_screenshot(filename)
        print(f"Screenshot saved: {filename}")
        return True
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return False


def move_map(driver, canvas, offset):
    ActionChains(driver).move_to_element_with_offset(canvas, 0, 0) \
        .click_and_hold() \
        .move_by_offset(*offset) \
        .release() \
        .perform()

def move_many(driver, canvas, offset, factor):
    if factor < 0:
        offset = (-1 * offset[0], -1 * offset[1])
    factor = abs(factor)

    for k in range(factor):
        move_map(driver, canvas, offset)


def reset_position(driver, lat, long, zoom):
    # Go to the first coordinate
    enter_coordinates_in_geocoder(driver, lat, long)
    time.sleep(1)
    zoom_click(driver, zoom)
    time.sleep(1)
    clear_geocoder_input(driver.find_element(By.CLASS_NAME, 'mapboxgl-ctrl-geocoder--input'))


def setup_driver():
    # Set up Chrome options (headless for automation)
    chrome_options = Options()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--start-maximized')

    # Path to chromedriver (assumes it's in your PATH)
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    url = os.getenv("FA_URL")
    driver.get(url)
    print(f"Accessed {url}")
    # Wait for the page to load
    print(f"Page title: {driver.title}")

    # User input
    time.sleep(1)

    # Enter password using the standalone function
    if enter_password(driver, os.getenv("FA_PASSWORD")):
        print("Password entered successfully.")

    # Find and click the submit button
    time.sleep(0.5)  # Small delay before clicking submit
    click_submit_button(driver)
    time.sleep(1)
    # Find and click the element containing "Double"
    click_element_by_text(driver, "Double")
    time.sleep(0.5)

    # Find and click the element containing "Single"
    click_element_by_text(driver, "Single")
    time.sleep(0.5)

    return driver

def screenshot_idx_list(driver, idx_list_path, tag):
    import csv
    # Load and parse idx_list file (CSV with i, j pairs)
    with open(idx_list_path, 'r') as f:
        reader = csv.reader(f)
        coords = [(int(float(row[0])), int(float(row[1]))) for row in reader]

    # Sort by i, then j for efficient traversal
    coords.sort()

    if not coords:
        return

    # Go to the first coordinate
    i0, j0 = 0, 0
    lat = 31.590170
    lon = 34.491024
    zoom_to = 12
    reset_position(driver, lat, lon, zoom_to)
    canvas = driver.find_element(By.CLASS_NAME, 'mapboxgl-canvas')

    j_offset = (-400, -340)
    i_offset = (340, -400)
    i_min = 6

    current_i, current_j = i0, j0

    for (i, j) in coords:
        if i < i_min:
            continue
        di = i - current_i
        if di != 0:
            move_many(driver, canvas, i_offset, di)
            current_i = i
        dj = j - current_j
        if dj != 0:
            move_many(driver, canvas, j_offset, dj)
            current_j = j
        time.sleep(2)
        take_screenshot(driver, i, j, tag)

def screenshot_all(driver, tag):
    # Go to initial coordinates and zoom
    lat = 31.590170
    lon = 34.491024
    zoom_to = 12

    reset_position(driver, lat, lon, zoom_to)

    j_offset = (-400, -340)
    i_offset = (340, -400)

    j_num = 26
    i_num = 85

    i_min = 0

    # Click and drag by 100 pixels in the minus x plus y direction
    canvas = driver.find_element(By.CLASS_NAME, 'mapboxgl-canvas')
    for i in range(i_num):
        if i >= i_min:
            take_screenshot(driver, i, 0, tag)
            for j in range(1, j_num):
                move_map(driver, canvas, j_offset)
                print(f"Dragged to i: {i}, j: {j}")
                # Take screenshot after drag
                take_screenshot(driver, i, j, tag)

            move_many(driver, canvas, j_offset, -j_num)

        move_map(driver, canvas, i_offset)


def setup_feat(driver):
    click_element_by_text(driver, "Rasters")
    return "feat"

def setup_label(driver):
    click_element_by_text(driver, "Tents")
    return "label"

if __name__ == "__main__":
    driver = setup_driver()
    RASTER_LAYER = True

    # Find and click the element containing "Rasters"
    if RASTER_LAYER:
        tag = setup_feat(driver)
        time.sleep(1)
        screenshot_idx_list(driver, "../idcs.csv", tag)
    else:
        tag = setup_label(driver)
        time.sleep(1)
        screenshot_all(driver, tag)
    driver.quit()