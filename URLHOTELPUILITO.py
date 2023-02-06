import sys
import csv
from selenium import webdriver
import time

url = 'https://www.tripadvisor.it/Hotels-g194949-Villasimius_Province_of_Cagliari_Sardinia-Hotels.html'
driver = webdriver.Chrome(executable_path='C:\\Users\\39339\\Downloads\\chromedriver_win32 (2)\\chromedriver')
driver.get(url)
time.sleep(2)                                   # Fondamentale per far cliccare sul tasto ok
driver.find_element_by_xpath('//button[@class="evidon-banner-acceptbutton"]').click()           # CLICCA OK

path_to_file = "/Users/39339/Desktop/urlHotel.csv"
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['url'])

lista_url = []
numPag = 5

for n in range(0, numPag):

    container = driver.find_elements_by_xpath(".//div[@class='prw_rup prw_meta_hsx_responsive_listing ui_section listItem']")

    for n in range(0, len(container)):
        url = container[n].find_element_by_xpath(".//a[contains(@href, '/Hotel_Review')]").get_attribute('href')

        if url not in lista_url:
            lista_url.append(url)
            csvWriter.writerow([url])

    time.sleep(2)
    try:
        driver.find_element_by_xpath("//a[@class='nav next ui_button primary']").click()  # NEXT BUTTON
        time.sleep(2)                       # Fondamentale per fare il cambio pagina
    except:
        print("Unexpected error:", sys.exc_info()[0])

    time.sleep(2)                                                 #tempo di riposo tra un url e l'altra
driver.close()