import random
import nltk
import pandas as pd
from nltk import bigrams




import random
import nltk
import pandas as pd
from nltk import bigrams
from sklearn.model_selection import train_test_split

lista_ds = ["../Topic/recensioni_attrazioni_complete.csv", "../Topic/recensioni_hotel_complete.csv", "../Topic/recensioni_ristoranti_complete.csv"]
for file_csv in lista_ds:
    ds = pd.read_csv(file_csv)
    lista_rec = []
    testo_recensioni = ''
    for index, review in enumerate(ds.Recensione_pulita):
        testo_recensioni += review + ' '
        lista_rec.append((review.split(), ds.Giudizio[index]))

    text_no_stopwords = testo_recensioni.split()  # and w == 'non' or......# ELIMINIAMO LE STOPWORDS
    bigramma = bigrams(text_no_stopwords)

    all_words_no_stop = nltk.FreqDist(w for w in bigramma)
    frequenze= all_words_no_stop.keys()
    print(all_words_no_stop)
    sample = len(frequenze)
    n_features_rilevanti = [int(sample*0.01),int(sample*0.015),int(sample*0.02)] #per decidere il valore di features migliore

    for n in n_features_rilevanti:
        print(n)
        word_features_no_stop = list(all_words_no_stop)[:n]

        reviews_bigrams = []
        for recensione in lista_rec:
            bigram = bigrams(recensione[0])
            reviews_bigrams.append(
                (bigram, recensione[1]))  # recensione[0] = recensione in bigrammi, recensione[1] = giudizio


        def review_features(review, bigramma_features):
            document_bigrams = set(review)
            features = {}
            for bigram in bigramma_features:
                features[bigram] = bigram in document_bigrams
            return features

        feature_sets = [(review_features(review, word_features_no_stop), c) for (review, c) in
                        reviews_bigrams]

        X_train, X_test, y_train, y_test = train_test_split(feature_sets, ds.Giudizio, test_size=0.3, shuffle=True, random_state=42, stratify=ds.Giudizio)
        classifier = nltk.NaiveBayesClassifier.train(X_train)
        print(nltk.classify.accuracy(classifier, X_test))
        classifier.show_most_informative_features(50)
