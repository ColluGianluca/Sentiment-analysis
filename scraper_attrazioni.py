import csv
#import sys
import pandas as pd
from selenium import webdriver
import time

path_to_file = "/Users/39339/Desktop/Recensioni_Attrazioni.csv"
dataset_url= pd.read_csv('urlAttrazioni.csv')
listaURL=dataset_url['url']

csvFile = open(path_to_file,'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['Attrazione','Data', 'Rating', 'Titolo', 'Recensione','Tipologia'])
driver = webdriver.Chrome(executable_path='..\\Chromedriver\\chromedriver')
for url in listaURL:
    driver.get(url)
    time.sleep(2)
    try:
        driver.find_element_by_xpath("//button[@id='_evidon-accept-button']").click()     # CLICK OK per accettare condizioni
    except:
         pass            # TRY EXCEPT NECESSARIO : ALLA PRIMA URL CLICCA SU OK, LE VOLTE SUCCESSIVE LA PAGINA E' GIA APERTA
    time.sleep(2)
    try:
            try:
                attrazione=driver.find_element_by_xpath("//h1[@class='DrjyGw-P _1SRa-qNz qf3QTY0F']").text         #PRENDE IL NOME DELL'ATTRAZIONE
                try:
                    a = driver.find_element_by_xpath(".//div[@class='_1NyglzPL']").text       # NUMERO TOTALE RECENSIONI IN ITALIANO
                    if a != '':
                        Num_Recensioni_In_Italiano=(a.split()[2])
                        if len(Num_Recensioni_In_Italiano)>3:
                            Num_Recensioni_In_Italiano=float(Num_Recensioni_In_Italiano)*1000 # NECESSARIO PERCHE' SE SONO PRESENTI LE MIGLIAIA NEL NUMERO RECENSIONI ALLORA E' PRESENTE UN PUNTO SEPARATORE
                except:
                   print('è presente una sola pagina')                  # scritta presente diverse volte, eccezione importante
                   Num_Recensioni_In_Italiano=0
                numeroPagine=int(float(Num_Recensioni_In_Italiano)/6)+1                 # ABBIAMO PRESO IL NUMERO TOTALE DELLE RECENSIONI IN ITALIANO, ABBIAMO DIVISO IL NUMERO PER 6 (VENGONO VISUALIZZATE SEMPRE 6 RECENSIONI A PAGINA) E POI AGGIUNGIAMO 1 ARROTONDANDO PER ECCESSO

                url1 = driver.find_element_by_xpath(".//a[contains(@href, '/ShowUserReviews')]").get_attribute('href')   # OTTENIAMO LA NUOVA URL PER SCRAPING ATTRAZIONE (ALTRIMENTI NON POSSIBILE, HTML COMPLETAMENTE DIVERSO)
                driver.close()         # CHIUDIAMO PAGINA DOPO AVER ESTRAPOLATO URL
                driver = webdriver.Chrome(executable_path='..\\Chromedriver\\chromedriver')
                driver.get(url1)                                            # RIAPRIAMO IL DRIVER CON LA NUOVA URL
                time.sleep(2)
                driver.find_element_by_xpath("//button[@id='_evidon-accept-button']").click()     #CLICK OK su accetta condizioni
                time.sleep(2)

                for i in range(0,numeroPagine):
                    time.sleep(2)
                    container = driver.find_elements_by_xpath(".//div[@class='review-container']")
                    for j in range(len(container)):
                        try:
                            name = attrazione
                            title = container[j].find_element_by_xpath(".//span[@class='noQuotes']").text
                            date = container[j].find_element_by_xpath(".//span[contains(@class, 'ratingDate')]").get_attribute("title")
                            rating = \
                                container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute(
                                "class").split("_")[3]
                            review = container[j].find_element_by_xpath(".//p[@class='partial_entry']").text.replace("\n", " ")
                            tipologia= 'spiaggia'
                            csvWriter.writerow([name, date, rating, title, review,tipologia])
                        except:
                            print('dato non disponibile')
                    try:
                        driver.find_element_by_xpath('.//a[@class="nav next ui_button primary"]').click()             # CAMBIA PAGINA
                    except:
                        pass
            except:

                attrazione = driver.find_element_by_xpath("//h1[@id='HEADING']").text    #PRENDE IL NOME DELL'ATTRAZIONE

                recens_container = driver.find_element_by_xpath(".//label[@for='LanguageFilter_1']")    # NUMERO TOTALE RECENSIONI IN ITALIANO
                a = recens_container.find_element_by_xpath('.//span[@class="mxlinKbW"]').text
                Num_Recensioni_In_Italiano = float(''.join([i for i in a if i.isnumeric()]))

                numeroPagine = int(float(
                    Num_Recensioni_In_Italiano) / 5) + 1  # ABBIAMO PRESO IL NUMERO TOTALE DELLE RECENSIONI IN ITALIANO, ABBIAMO DIVISO IL NUMERO PER 5 ( VENGONO VISUALIZZATE SEMPRE 5 RECENSIONI A PAGINA) E POI AGGIUNGIAMO 1 PERCHE' COSI' ARROTONDIAMO PER ECCESSO

                for i in range(0, numeroPagine):
                    try:
                        time.sleep(2)
                        driver.find_element_by_xpath("//span[@class='_3maEfNCR']").click()  # CLICCA 'LEGGI DI PIU' NELLE RECENSIONI
                    except:
                        pass            #errore: nessun 'leggi di più' da cliccare'
                    time.sleep(2)
                    container = driver.find_elements_by_xpath(".//div[@class='oETBfkHU']") #LISTA DEI CONTAINER DELLE RECENSIONI
                    for j in range(len(container)):
                        name = attrazione
                        title = container[j].find_element_by_xpath(".//a[@class='ocfR3SKN']").text
                        try:
                            date= container[j].find_element_by_xpath(".//span[@class='_34Xs-BQm']").text.replace("Data dell'esperienza: ", "")
                        except:
                            date= 'non disponibile'
                        rating = \
                            container[j].find_element_by_xpath(
                                ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute(
                                "class").split("_")[3]
                        review = container[j].find_element_by_xpath(".//q[@class ='IRsGHoPm']").text.replace("/n", " ")
                        tipologia= 'attività'
                        csvWriter.writerow([name, date, rating, title, review,tipologia])
                    try:
                        driver.find_element_by_xpath('.//a[@class="ui_button nav next primary "]').click()  # CAMBIA PAGINA
                    except:
                        pass  # non cambia pagina

    except:                     # due strutture sono sottocategorie di attrazioni, except utile
        pass                    # L'ATTRAZIONE NON E' STATA TROVATA (BIKE GREEN e TESSUTI RUBRUM). ESSE FANNO PARTE DI UNA SOTTOCATEGORIA, CON DIVERSO CODICE HTML- tag e attributi diversi-
                                # IN OGNI CASO HANNO ENTRAMBE 1 RECENSIONE, RISULTANO QUINDI ININFLUENTI AI FINI DELL'ANALISI, ABBIAMO PREFERITO NON APPESANTIRE ULTERIORMENTE IL CODICE...
driver.close()
csvFile.close()

ds = pd.read_csv("./recensioni_attrazioni.csv")
ds['Giudizio'] = ''
for index,i in enumerate(ds['Rating']):
    if int(i) in [10,20]:
        ds['Giudizio'][index] += 'Negative'
    if int(i) in [40,50]:
        ds['Giudizio'][index] += 'Positive'
    if int(i) == 30:
        ds['Giudizio'][index] += 'Neutral'

ds.to_csv("../Sentiment/recensioni_attrazioni_giudizi.csv", index=False)