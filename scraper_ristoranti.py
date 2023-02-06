import csv
import sys
from selenium import webdriver
import pandas as pd
import time

path_to_file = "./recensioni_ristoranti.csv"
csvFile = open(path_to_file,'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['Ristorante','Data', 'Rating', 'Titolo', 'Recensione'])
driver = webdriver.Chrome(executable_path='..\\Chromedriver\\chromedriver')

dataset_url= pd.read_csv('../Url/url_ristoranti.csv')
listaURL=dataset_url['url']

for url in listaURL:
    driver.get(url)
    try:
        driver.find_element_by_xpath("//button[@id='_evidon-accept-button']").click()     # Clicca l'OK per accettare le condizioni
    except:
        pass
    container = driver.find_elements_by_xpath(".//label[@for='filters_detail_language_filterLang_it']")     # selezioniamo l'icona italiano in 'Lingue'
    restaurant=driver.find_element_by_xpath(".//h1[@class='_3a1XQ88S']").text                               # salviamo il nome del ristorante
    for n in range(0,len(container)):
         a=container[n].find_element_by_xpath(".//span[@class='count']").text                               # otteniamo il numero di recensioni in italiano
         if a !='':
             break
    Num_Recensioni_In_Italiano = float(''.join([i for i in a if i.isnumeric()]))
    if Num_Recensioni_In_Italiano > 10:
        numero_pagine = Num_Recensioni_In_Italiano / 10
        if (int(str(numero_pagine)[-1]) < 5):
            num_page = int(round(numero_pagine, 0)) + 1
        else:
            num_page = int(round(numero_pagine, 0))

    if int(Num_Recensioni_In_Italiano) <= 10:
        num_page = 1

    for i in range(0, num_page):
        try: 				# non strettamente necessario
            time.sleep(2)
            driver.find_element_by_xpath("//span[@class='taLnk ulBlueLinks']").click()     #clicca su espandi la recensione, se presente
            time.sleep(3)
        except:
            pass

        container = driver.find_elements_by_xpath(".//div[@class='review-container']")     #identifichiamo i containers delle singole reviews

        for j in range(len(container)):
            try:			#da verificarne la necessità
                name=restaurant
                title = container[j].find_element_by_xpath(".//span[@class='noQuotes']").text
                date = container[j].find_element_by_xpath(".//span[contains(@class, 'ratingDate')]").get_attribute("title")
                rating = \
                container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute(
                    "class").split("_")[3]
                review = container[j].find_element_by_xpath(".//p[@class='partial_entry']").text.replace("\n", " ")
                csvWriter.writerow([name, date, rating, title, review])
            except:
                pass

            try:
                driver.find_element_by_xpath('.//a[@class="nav next ui_button primary"]').click()     #clicca su cambio pagina, se presente
            except:
                pass

driver.close()
csvFile.close()

ds = pd.read_csv("./recensioni_ristoranti.csv")
ds['Giudizio'] = ''
for index,i in enumerate(ds['Rating']):
    if int(i) in [10,20]:
        ds['Giudizio'][index] += 'Negative'
    if int(i) in [40,50]:
        ds['Giudizio'][index] += 'Positive'
    if int(i) == 30:
        ds['Giudizio'][index] += 'Neutral'

ds.to_csv("../Sentiment/recensioni_ristoranti_giudizi.csv", index=False)