class Loader():

    def __init__(self,filename,path):
      self.filename,self.path=filename,path

    def load(self):
        df = spark.read.csv(os.path.join(self.path, self.filename), header=True, inferSchema=True)
        df=df.dropna()
        print(f"The dataset contains :{df.count()} observation")
        return df

class Pre_index():

    def __init__(self,index_dataset):
        self.dataset=index_dataset

    def Indexer(self,target):
        variable_to_index = [col for col in list(set(self.dataset.columns)) if
                          dict(self.dataset.dtypes)[col] == "string"]
        variable_indexed=[col+'_I' for col in list(set(self.dataset.columns)) if
                          dict(self.dataset.dtypes)[col] == "string"and col!=target]

        return variable_to_index, variable_indexed

class classificatore(object):

        def __init__(self,scaled_dataframe,target):
            self.dataframe=scaled_dataframe
            self.target=target
            self.train_set,self.test_set= self.dataframe.randomSplit([0.7,0.3], seed=0)

        def Logistic_Regression(self):
            regressor = LogisticRegressor(featuresCol="scaledFeatures", labelCol=self.target,threshold=0.27)
            model = regressor.fit(self.train_set)
            performance = model.evaluate(self.test_set)
            predictions = performance.predictions.select(self.target, 'prediction')
            evaluator = BinaryEvaluator(rawPredictionCol='prediction', labelCol=self.target)
            accuracy = evaluator.evaluate(predictions)
            Sens_spec = self.Confusion_matrix(predictions)
            print(f"Logistic Regression Accuracy: {accuracy:5.2}\n"f"Sensitivity : {Sens_spec[0]:5.2}\n"f" Specificity :{Sens_spec[1]:5.2}")

        def Random_Forest(self):
            classifier = RF(maxDepth=10, numTrees=15, labelCol=self.target, maxBins=50,featuresCol="scaledFeatures")
            model = classifier.fit(self.train_set)
            self.predictions = model.transform(self.test_set)
            self.tester = Tester(predictionCol='prediction', labelCol=self.target)
            accuracy = self.tester.evaluate(self.predictions)
            Sens_spec = self.Confusion_matrix(self.predictions)
            print(f"Random Forest Accuracy: {accuracy:5.2}\n"f"Sensitivity : {Sens_spec[0]:5.2}\n"f"Specificity :{Sens_spec[1]:5.2}")

        def Random_Forest_threshold(self):
            probabilities = [row['probability'][1] for row in self.predictions.collect()]
            new_predictions = [0 if i < 0.30 else 1 for i in probabilities]
            y_true = [row[self.target] for row in self.predictions.collect()]
            predictions = spark.createDataFrame(zip(new_predictions, y_true), schema=['prediction', self.target])
            ## TRASFORMIANO LE VARIABILI IN DOUBLE TYPE PER POTER VALUTARE L'ACCURACY
            predictions = predictions.withColumn("prediction", predictions.prediction.cast(DoubleType()))
            predictions = predictions.withColumn(self.target, predictions.salary_I.cast(DoubleType()))
            accuracy = self.tester.evaluate(predictions)
            Sens_spec = self.Confusion_matrix(predictions)
            print(f"Random Forest Threshold Accuracy: {accuracy:5.2}\n"f"Sensitivity : {Sens_spec[0]:5.2}\n"f"Specificity :{Sens_spec[1]:5.2}")

        def resampling_data(self):
            major_df = self.dataframe.filter(self.dataframe[self.target] == 0)
            minor_df = self.dataframe.filter(self.dataframe[self.target] == 1)
            oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in range(0,2)]))).drop('dummy')
            ## DATASET UNION
            combined_df = major_df.unionAll(oversampled_df)
            return combined_df

        def Multilayer_Perceptron(self):
            training_set_resampled, test_set_resampled =self.resampling_data().randomSplit([0.7, 0.3], seed=0)
            first_row=self.dataframe.collect()[1]
            features= [features for features in first_row[0]]
            layers = [len(features),20,10,2]
            trainer = MultilayerPerceptronClassifier(maxIter=100,layers=layers, blockSize=100, # stepSize=0.1, tol=0.0001, risultati migliori ma molto molto lento ( sens:93, spec:62, acc: 85)
                                                     seed=0,labelCol='salary_I',featuresCol='scaledFeatures')
            model = trainer.fit(training_set_resampled)
            predictions = model.transform(test_set_resampled)
            evaluator = Tester(labelCol=self.target, predictionCol='prediction',metricName='accuracy')
            accuracy=evaluator.evaluate(predictions)
            Sens_spec=self.Confusion_matrix(predictions)
            print (f"Multilayer Perfprmance Accuracy: {accuracy:5.2f}\n"f"Sensitivity : {Sens_spec[0]:5.2f}\n"f" Specificity : {Sens_spec[1]:5.2f}")

        def Confusion_matrix(self,pred):
            pred=pred.withColumnRenamed(self.target,'target')
            True_pos = pred.filter((pred.target ==0) & (pred.prediction==0)) # TRUE POSITIVE
            False_pos = pred.filter((pred.target ==1)& (pred.prediction==0))    # FALSE POSITIVE
            False_neg = pred.filter((pred.target ==0)& (pred.prediction==1))    # FALSE NEGATIVE
            True_neg = pred.filter((pred.target == 1) & (pred.prediction == 1))  # TRUE NEGATIVE

            self.Sensibility= True_pos.count()/ (False_neg.count()+True_pos.count())  # tasso veri positvi predetti (>50K)
            self.Specificity= True_neg.count()/ (True_neg.count()+False_pos.count()) # tasso di veri nengativi predetti(<=50K)
            return self.Sensibility,self.Specificity,

class Regression():
    def __init__(self,dataset):
        self.dataset=dataset

    def LRegr(self):
        training_set, test_set = self.dataset.randomSplit([0.8, 0.2])
        regressor = LinearRegressor(featuresCol='scaledFeatures', labelCol='price', predictionCol='prediction',regParam=0.3)
        model = regressor.fit(training_set)
        test_result = model.evaluate(test_set)
        print('RMSE:{}'.format(test_result.rootMeanSquaredError))
        print('MSE:{}'.format(test_result.meanSquaredError))
        print('R2:{}'.format(model.summary.r2))

if __name__ == '__main__':
    from pyspark.sql.types import IntegerType, DoubleType
    from pyspark.ml.classification import LogisticRegression as LogisticRegressor, RandomForestClassifier as RF,MultilayerPerceptronClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator as BinaryEvaluator,RegressionEvaluator,MulticlassClassificationEvaluator as Tester
    from pyspark.ml.regression import LinearRegression as LinearRegressor
    from pyspark.sql.functions import col, explode, array, lit
    from pyspark.ml.feature import VectorAssembler,MinMaxScaler,RobustScaler,StringIndexer
    from pyspark.ml import Pipeline; import  category_encoders as ce
    import pyspark;            import warnings ;                 warnings.filterwarnings('ignore')
    import os;                 import sys ;                      from pyspark.sql import SparkSession
    import pandas as pd;       import findspark ; findspark.init()

    spark = SparkSession.builder.appName("logistica").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('WARN')
    biglab = os.environ['BIGLAB']; datapath = os.path.join(biglab, "datasets")

    ####### SALARY DATASET ######
    #
    # dname = 'salary.csv'
    # # salary=Loader(dname,datapath)
    # # salary_data_complete=salary.load()
    # pre_index=Pre_index(salary_data_complete)
    # get_variables=pre_index.Indexer('salary')
    #
    # numeric_column = [col for col in salary_data_complete.columns if col not in get_variables[0]]
    # features = get_variables[1] + numeric_column
    # ## COSTRUIAMO LA PIPELINE
    # Indexed=StringIndexer(inputCols=get_variables[0], outputCols=[col + "_I" for col in get_variables[0]])
    # assembler= VectorAssembler (inputCols= features,outputCol='features')
    # scaler= MinMaxScaler(inputCol='features',outputCol='scaledFeatures')
    # pipeline=Pipeline(stages=[Indexed,assembler,scaler])
    # Scaled_data=pipeline.fit(salary_data_complete).transform(salary_data_complete).select('scaledFeatures','salary_I')
    # classifier=classificatore(Scaled_data,'salary_I')

    ## ESEGUIAMO I CLASSIFICATORI ##
    # classifier.Logistic_Regression()
    # classifier.Random_Forest()
    # classifier.Random_Forest_threshold()
    # classifier.Multilayer_Perceptron()
    ################# REGRESSION #########

    def Cleaning(df):
        data = df.withColumnRenamed('neighbourhood_group', 'area') \
            .withColumnRenamed('neighbourhood', 'sub-area') \
            .withColumnRenamed('number_of_reviews', 'reviews_number') \
            .withColumnRenamed('monthly_rev', 'rev_per_month') \
            .withColumnRenamed('minimum_nights', 'min_nights') \
            .withColumnRenamed('calculated_host_listings_count', 'host_number') \
            .withColumnRenamed('availability_365;;;;;', 'availability365')

        dataframe = data.withColumn('min_nights', data.min_nights.cast(IntegerType()))
        dataframe = dataframe.withColumn('price', dataframe.price.cast(IntegerType()))
        dataframe = dataframe.withColumn('reviews_number', dataframe.reviews_number.cast(IntegerType()))
        dataframe = dataframe.withColumn('reviews_per_month', dataframe.reviews_per_month.cast(IntegerType()))
        df = dataframe.withColumn('host_number', dataframe.host_number.cast(IntegerType()))
        return df.drop('availability365','last_review')
    #
    # ## DECOMMENTARE LE SUCCESSIVE RIGHE PER L'OUTPUT REGRESSION
    dname2 = 'AB_NYC_2019.csv'
    ny = Loader(dname2, datapath)
    ny_data = ny.load()
    filter_data=ny_data.filter((ny_data.price > 0) & (ny_data.price < 1000))
    ny_data=Cleaning(filter_data)
    values = ['Private room', 'Entire home/apt', 'Shared room']
    data = ny_data.filter(ny_data.room_type.isin(values))

    pre_index=Pre_index(data)
    get_variable=pre_index.Indexer('price')
    ## ORDINAL ENCODER ##
    encoder = ce.OrdinalEncoder(cols=['room_type', 'area'], return_df=True,
                                    mapping=[{'col': 'room_type',
                                              'mapping': {'Shared room': 0, 'Private room': 1, 'Entire home/apt': 2}},
                                             {'col': 'area',
                                              'mapping': {'Bronx': 0, 'Queens': 1, 'Staten Island': 2, 'Brooklyn': 3,
                                                          'Manhattan': 4}}])
    dataframe = ny_data.toPandas()
    dataframe = encoder.fit_transform(dataframe)
    dataframe = spark.createDataFrame(dataframe)


    numeric_column = [col for col in dataframe.columns if col not in get_variable[0] and col!= 'price']
    features = get_variable[1] + numeric_column
    ## COSTRUIAMO LA PIPELINE
    Indexed = StringIndexer(inputCols=get_variable[0], outputCols=[col + "_I" for col in get_variable[0]])
    assembler = VectorAssembler(inputCols=features,outputCol='features')
    scaler=RobustScaler(inputCol='features',outputCol='scaledFeatures')
    pipeline=Pipeline(stages=[Indexed,assembler,scaler])
    Scaled_data = pipeline.fit(dataframe).transform(dataframe)
    ## MODELLO DI REGRESSIONE
    regressione=Regression(Scaled_data)
    regressione.LRegr()
