import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import GridSearchCV


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]  + ','
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


def LDA(df):

    dataset=pd.read_csv(df)
    most_reviews = []
    negative = []
    negative_most_reviewed=[]
    positive_most_reviewed = []
    positive = []
    name=list(dataset)[0]
    print(name)
    for nome in set(dataset.iloc[:, 0]):
        count = 0
        for i in (dataset.iloc[:, 0]):
            if nome == i:
                count += 1
                if count == 1000:
                    most_reviews.append(i)

    for index, giudizio in enumerate(dataset['Giudizio_SVM']):

        if giudizio == 'Negative':

            if dataset.iloc[:, 0][index] not in most_reviews:
             negative.append(dataset.Recensione_pulita[index])          # PRENDE LE NEGATIVE DEI LUOGHI MENO RECENSITI
            if dataset.iloc[:, 0][index] in most_reviews:
                negative_most_reviewed.append(dataset.Recensione_pulita[index])

        if giudizio == 'Positive':
            if dataset.iloc[:, 0][index] not in most_reviews:
             positive.append(dataset.Recensione_pulita[index])          # PRENDE LE POSITIVE DEI LUOGHI MENO RECENSITI
            if dataset.iloc[:, 0][index] in most_reviews:
                positive_most_reviewed.append(dataset.Recensione_pulita[index])

    print('topic delle recensioni negative di %s meno recensiti: ' % name)
    get_topic(negative)
    print('topic delle recensioni negative di %s più recensiti: ' % name)
    if len(negative_most_reviewed) != 0:
        get_topic(negative_most_reviewed)
    else:
        print('Assenti')
    print('topic delle recensioni positive di %s meno recensiti: ' % name)
    get_topic(positive)
    print('topic delle recensioni positive di %s più recensiti: ' % name)
    if len(positive_most_reviewed) != 0:
        get_topic(positive_most_reviewed)
    else:
        print('Assenti')

def get_topic(dati=[]):

    documents = dati
    no_top_words = 10
    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    min_df=2,
                                    ngram_range=(2,2),
                                      )
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
        #OTTIMIZZAZIONE DEI PARAMETRI:
    lda=LatentDirichletAllocation()
    searchParams = {'n_components': [5,10,15], 'learning_decay': [.5, .7, .9]}
    model = GridSearchCV(lda, param_grid=searchParams)
    model.fit(tf)
    best_parameters = model.best_estimator_.get_params()
    print(best_parameters)
    print(model.best_score_)
    lda = LatentDirichletAllocation(
            n_components=best_parameters['n_components'],
            max_iter=best_parameters['max_iter'],
            learning_method=best_parameters['learning_method'],
            learning_decay=best_parameters['learning_decay'],
            learning_offset=best_parameters['learning_offset'],
            batch_size=best_parameters['batch_size'],
            max_doc_update_iter=best_parameters['max_doc_update_iter'],

            ).fit(tf)

    display_topics(lda, tf_feature_names, no_top_words)

if __name__ == '__main__':

    file=['./Topic/solo_attività.csv','./Topic/solo_spiagge.csv','./Topic/recensioni_ristoranti_complete.csv','./Topic/recensioni_hotel_complete.csv']
    for ds in file:
        print('\n')
        LDA(ds)
