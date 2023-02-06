import nltk
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


lista_ds = ["./recensioni_attrazioni_complete.csv","./recensioni_hotel_complete.csv"] #,"./recensioni_ristoranti_complete.csv"
for file_csv in lista_ds:
    dataset = pd.read_csv(file_csv)

    giudizi = ['Negative', 'Neutral', 'Positive']
    x = dataset.Recensione_pulita
    y = dataset.Giudizio
    documents = x
    testo = ''
    for doc in documents:
        testo += doc + ' '
    all_words = [words for words in testo.split()]
    frequenze = nltk.FreqDist(all_words).keys()
    sample = (len(frequenze))

    n_features = int(int(sample)*0.25)    #se si desidera inserire il parametro max_features=n_features di TfidfVectorizer
    print(sample,n_features)

    # }
    parameters = {        # 2 codice utilizzato funzionante
        'vect__max_df': [0.90,0.95],
        'vect__max_features': [sample, n_features],
        'vect__ngram_range': [[1, 2]],  # unigrams or bigrams
        'clf__kernel': ['linear','rbf'],
        'clf__C': [1,10],
        'clf__probability': [True],
        'clf__class_weight': ['balanced']
    }
    classifier = Pipeline([
        ('vect', TfidfVectorizer(strip_accents='unicode',
                                 tokenizer=word_tokenize,
                                 decode_error='ignore',
                                 analyzer='word',
                                 norm='l2'
                                 )),
        ('clf', SVC(shrinking=True))
    ])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)
                                       # utilizziamo il parametro stratify per mantenere le proporzioni tra le classi sia nel train che nel test set
    grid_search = GridSearchCV(classifier, parameters, verbose=1,scoring='f1_weighted')
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in classifier.steps])
    print("parameters:")
    print(parameters)
    grid_search.fit(X_train, y_train)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    predicted = grid_search.predict(X_test)
    print(accuracy_score(y_test, predicted))
    #print(precision_recall_fscore_support(y_test, predicted))
    print(classification_report(y_test, predicted))


    #Aggiunta della colonna predizione del miglior classificatore SVM ai dataset
    y_finale = grid_search.predict(x)
    print(len(y_finale))
    dataset.insert(10, 'Giudizio_SVM', y_finale, allow_duplicates=False)
    dataset.to_csv(file_csv, index=False)


