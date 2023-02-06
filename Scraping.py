import sys
import csv
# from telnetlib import EC

from selenium import webdriver
import time

# default path to file to store data
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.wait import WebDriverWait

path_to_file = "/Users/39339/Desktop/reviews.csv"

# default number of scraped pages
# num_page = 1
# # Import the webdriver
url = "https://www.tripadvisor.it/Restaurant_Review-g194949-d15137345-Reviews-Gnocco-Villasimius_Province_of_Cagliari_Sardinia.html"
driver = webdriver.Chrome(executable_path='C:\\Users\\39339\\Downloads\\chromedriver_win32 (2)\\chromedriver')
driver.get(url)
container = driver.find_elements_by_xpath(".//label[@for='filters_detail_language_filterLang_it']")    # PRENDE L'ICONa SOTTO TUTTE LE LINGUE. pRENDE LA CASELLA ITALIANO E ANNESSO NUMERO

for n in range(0,len(container)):
 a=container[n].find_element_by_xpath(".//span[@class='count']").text                        # GRAZIE AL CICLO FOR OTTENIIAMO IL NUMERO DI RECENSIONI IN ITALIANO
 if a !='':
     Num_Recensioni_In_Italiano=a
     break
nuove=[num for num in Num_Recensioni_In_Italiano if num.isnumeric()]
numeroPagine= ''.join(nuove)
numeroPagine=float(numeroPagine)/10
if (int(str(numeroPagine)[-1]) < 5) :
    num_page=int(round(numeroPagine,0))
else:
    num_page = int(round(numeroPagine, 0)) -1
print(num_page)
#if you pass the inputs in the command line
#if (len(sys.argv) == 4):
   # path_to_file = sys.argv[1]
    #num_page = int(sys.argv[2])
    #url = sys.argv[3]

# container = driver.find_elements_by_xpath(".//label[@for='filters_detail_language_filterLang_it']")    # PRENDE L'ICONO SOTTO TUTTE LE LINGUE. pRENDE LA CASELLA ITALIANO E ANNESSO NUMERO

# for n in range(0,len(container)):
#  Recensioni_In_Italiano=container[n].find_element_by_xpath(".//span[@class='count']").text                        # GRAZIE AL CICLO FOR OTTENIIAMO IL NUMERO DI RECENSIONI IN ITALIANO
#  print(type(Recensioni_In_Italiano))
#
# Open the file to save the review
csvFile = open(path_to_file,'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['date', 'rating', 'title', 'review'])
# change the value inside the range to save more or less reviews
for i in range(0, num_page):

    # expand the review
    try:
     time.sleep(4)
     driver.find_element_by_xpath("//span[@class='taLnk ulBlueLinks']").click()
     time.sleep(2)
    except:
        print("Unexpected error:", sys.exc_info()[0])

    container = driver.find_elements_by_xpath(".//div[@class='review-container']")

    for j in range(len(container)):
        title = container[j].find_element_by_xpath(".//span[@class='noQuotes']").text
        date = container[j].find_element_by_xpath(".//span[contains(@class, 'ratingDate')]").get_attribute("title")
        rating = \
        container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute(
            "class").split("_")[3]
        review = container[j].find_element_by_xpath(".//p[@class='partial_entry']").text.replace("\n", " ")
        csvWriter.writerow([date, rating, title, review])

        # change the page
    driver.find_element_by_xpath('.//a[@class="nav next ui_button primary"]').click()

driver.close()