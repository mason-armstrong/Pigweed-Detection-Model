#Class to gather images from the web
from google_images_download import google_images_download
from bing_image_downloader import downloader
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
import pandas as pd
import time
import datetime as datetime

driver = webdriver.Firefox()  # or webdriver.Chrome()
inaturalist_directory = 'data/raw/inaturalist_images'
csv_link = "PIGWEED_IMAGES_GBIF_CLEANED.csv"

# Parse csv for image links
def parse_csv(csv_path):
    url_list = []
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        url = row['occurrenceID']
        # Check that the url is valid
        if url.startswith("http"):
            url_list.append(url)
    return url_list

#Download image from url
def download_image(url, directory):
    # Image name as timestamp to avoid duplicates
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"{directory}/temp_{timestamp}.png"
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img.save(save_path)
    

def download_image_from_inaturalist(driver, naturalist_link):
    # Navigate to the webpage
    driver.get(naturalist_link)
    timemstamp  = str(time.time())
    # Wait for the image to load
    parent_div = WebDriverWait(driver, 99999).until(
        EC.presence_of_element_located((By.CLASS_NAME, "image-gallery-image"))
    )
    try:
        image_elements = driver.find_elements(By.CLASS_NAME,"image-gallery-image")
       
        # Iterate through image links and download the images
        for i, image_element in enumerate(image_elements):
            image_tag = image_element.find_element(By.TAG_NAME, "img")
            image_src = image_tag.get_attribute("src")
            download_image(image_src, inaturalist_directory) 
        
    except Exception as e:
        print(f"An error occurred: {e}")

    
url_list = []
url_list = parse_csv(csv_path=csv_link)

for i, url in enumerate(url_list):
    # Sleep for 5 seconds every 40 images to avoid getting blocked
    if i % 20 == 0:
        time.sleep(5)
    download_image_from_inaturalist(driver, url)
    time.sleep(.1)
    print(f"Downloaded image {i:04d}")   
    
driver.close()

