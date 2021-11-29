from selenium import webdriver
import urllib.request

driver = webdriver.Firefox()

driver.get("https://www.geeksforgeeks.org/")



'''video_url = video.get_property('src')
urllib.request.urlretrieve(video_url, 'videoname.mp4')'''