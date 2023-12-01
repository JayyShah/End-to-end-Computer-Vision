from selenium import webdriver
import io, os, time
import requests
import hashlib
from PIL import Image

def scroll_to_end(mydriver, sleeptime):
    mydriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(sleeptime)


def get_image_urls(query , max_images_to_fetch, mydriver, sleeptime):

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    mydriver.get(search_url.format(q=query))

    image_urls = set()
    image_count,results_start  = 0,0

    while image_count < max_images_to_fetch:
        scroll_to_end(mydriver, sleeptime)

        # get all image thumbnail results
        results = mydriver.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(results)
        print("Found: {} search results. Extracting links from {}:{}".format(number_results,results_start,number_results))

        for img in results[results_start:number_results]:

            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleeptime)
            except Exception:
                print('ERROR while clicking')
                continue

            # extract image urls
            actual_images = mydriver.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)
            print("Found {} images".format(image_count))

            if len(image_urls) >= max_images_to_fetch:
                print("Found: {} image links. All done.".format(len(image_urls)))
                break

        else:
            print("Found: {} image links, looking for more ...".format(len(image_urls)))
            time.sleep(10)
            load_more_button = mydriver.find_element_by_css_selector(".mye4qd")

            if load_more_button:
                mydriver.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(results)

    return image_urls



def download_image(folder_path, url):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print("SUCCESS - saved {} at path: {}".format(url, file_path))
    except Exception as e:
        print("ERROR - Could not download/save {} because of the error: {}".format(url, e))



def search_and_download(query_item ,driver_path ,max_images_to_fetch=1):

    with webdriver.Chrome(executable_path=driver_path) as mydriver:
        res = get_image_urls(query_item, max_images_to_fetch, wd=mydriver, sleeptime=0.5)

    # download images from fetched urls
    for elem in res:
        download_image(target_folder,elem)


if __name__ == '__main__':

    queries = ['football', 'soccer', 'soccer kick']
    NUM_IMAGES = 10

    curr_dir = os.getcwd()
    driver_path = os.path.join(curr_dir, 'chromedriver')
    target_folder = os.path.join(curr_dir, 'scraped_images')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for query in queries:
        search_and_download(query_item = query,driver_path = driver_path, max_images_to_fetch = NUM_IMAGES)
