# Machine Learning Engineer Nanodegree
## Capstone Project
27septembre  
March, 2023

## I. Definition

### Project Overview and Statement
The project is related to data analysis in business. It provides demographic data for customers of a mail-order sales company and demographic information for the general population in Germany. The goal is to analyze data and create a customer segmentation report, that identifies the key features of the core customer base of the company. It also helps to target the company's marketing campaign for potential customers.

There are two main steps: 1. using unsupervised learning techniques for customer segmentation. 2. build a model to predict which individuals are most likely to convert into future customers for the company.

### Metrics
-The K-Means algorithm clusters data into n groups of equal variance by minimizing a criterion known as the inertia or within-cluster sum-of-squares error (SSE). 

-One of the most common evaluation metrics for binary classification problem is Area Under ROC Curve. It will be applied in supervised learning part.

## II. Analysis

### Data Exploration and Visualization (needs figures)

There are six data files provided by this project:

- `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
- `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- `DIAS Attributes - Values 2017.xlsx`: a detailed mapping of data values for each feature in alphabetical order.
- `DIAS Information Levels - Attributes 2017.xlsx`: a top-level list of attributes and descriptions, organized by informational category; .
- `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

The first two are for the customer segmentation by unsupervised learning . `Udacity_AZDIAS_052018.csv` is  about 1.1 G with 366 features. A lot of anormal data needs to be fix before further analysis. The two Excel spreadsheets provide more description of columns in the first data files.  ase on the  former analysis, the last two are used to predict potential customers by a supervised model.

All the data contain missing data and mismatch, so it is necessary to do data preprocessing. 

### Algorithms and Techniques
Since the complexity of  data, principal component analysis (PCA) technique is applied for dimensionality reduction.  K-Means method will be used to make customer Segmentation Report for unsupervised learning.  Elbow plot  identifies the best number of clusters for K-Means algorithm.

To build a prediction model for potential customers,  the performance of "DecisionTreeClassifier", "RandomForestClassifier",  "GradientBoostingClassifier", "AdaBoostClassifier" will be compared in Scikit-learn package. The best model will go through Grid-search and Cross-validation for best parameters.

### Benchmark
The highest score of  the prediction is 0.88403 so far in Kaggle Competition.  According to the other works,  the performance of "GradientBoostingClassifier" can be around 0.79.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

#### 1.Dtype warning 
When load the origin data, the warning is shown: DtypeWarning: Columns (_CAMEO_INTL_2015_, _CAMEO_DEUG_2015_) in all files have mixed types. X and XX values in them, that need to be converted to NANs.

#### 2.Missing values
273 out of 366 columns  in `Udacity_AZDIAS_052018.csv` have nan, and 273 out of 369 columns  in  `Udacity_CUSTOMERS_052018.csv` have nan.  After exploring the two Atrributes files,  there are some values marked as ['unknown','unknown / no main age detectable','no transactions known','no transaction known']. They need to be replaced into NaN by building an assosiated dictionary from Atrributes files.

The columns who have more than 30% of NaN will be dropped out in `Udacity_AZDIAS_052018.csv`, while more than 20% in `Udacity_CUSTOMERS_052018.csv`. The rows who contain more than 50% of NaN will be dropped as well.

#### 3.Clean and Encode with categorical data
There are still three categorical columns [_CAMEO_DEU_2015 _, _EINGEFUEGT_AM _,_OST_WEST_KZ_] in Azidas:

-_CAMEO_DEU_2015_   means  "New German CAMEO Typology established together with Call Credit in late 2015" and is a combination of number and alphabet.  It is dropped for simplification.  
-_EINGEFUEGT_AM_   indicates the date of data.  Only the year is kept here.
-_OST_WEST_KZ_  indicates 'flag indicating the former GDR/FRG', can easily encode into 0 and 1.

####  4.Filling NaN data
For both unsupervised and supervised models that I want to use in Sklearn,  there will be no NaN in data.  To simplify,  I try to fill NaN with mean values.

#### 5.Scaling data
To keep all the features equal generally, the data neeeds to be within similar range.  After searching in the Internet, MinMaxScaler seems more suitable than StandardScaler here.

### Implementation and Refinement

#### 1. Customer Segmentation Report

##### 1.1 Principal component analysis (PCA)
After data preprocessing,   Azdias and Customer datasets still have 332 features. Principal component analysis (PCA) technique is applied for dimensionality reduction.  In Fig.1,  blue bar is for explained variance of each principal component,  orange bar for accumulated explained variance. The first twos PCs only explains 0.098 and 0.06 of the total.  From Fig.2 we can see first 150 PCs contain most the information (89%). Therefore I set n_components=150 in PCA for simplification. 

##### 1.2 K-Means clustering
The K-Means algorithm clusters data into n groups of equal variance by minimizing a criterion known as the inertia or within-cluster sum-of-squares error (SSE). Elbow plot identifies the best number of clusters for K-Means algorithm. Fig. shows the relationship between cluster number and SSE. The score  decreases sharply  for the first 6 clusters,  and then continues to decrease with lower slope. I set 10 clusters for K-Means clustering. Then I apply K-Means to make segmentation of population and customers data. The distributions of population and customers in 10 clusters demonstrates in Fig. and their difference in Fig.

#### 2. Supervised Learning Model
In this part,  some binary classifiers ("DecisionTreeClassifier", "RandomForestClassifier",  "GradientBoostingClassifier", "AdaBoostClassifier" ) are used to predirct the potential customers.

Before training, similar transformation data preprocessing and cleaning are applied in `Udacity_MAILOUT_052018_TRAIN.csv`. "RESPONSE" in the dataset is the target which indicates who is customer. There is only 532 customers  (1.2%  in the total) in the Training data. I use _train_test_split_  from Scikit-learn to split Trian dataset into 80% training and 20% validation set , which is the most common setting in machine learning. 

Firstly  I use the four classifiers with default parameters to choose the best classifier. A receiver operating characteristic curve  (roc_auc) is set as the score.  Base on the performance on validation dataset and overfitting situation (Fig. ),  
GradientBoostingClassifier is the best model in this case. Its validation score increases up to 65%, while training score decreases up to 92%.  With Grid search and cross validation, the best parameters are chosen.

> GBC_classifier = GradientBoostingClassifier(random_state=0)
cv_sets = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
parameters = {'learning_rate':[0.001,0.01],
             'n_estimators' : [50,100,200],
             'max_depth':[3,5]}
GBC_obj = GridSearchCV(GBC_classifier, 
                             parameters, 
                             scoring = "roc_auc", 
                             cv = cv_sets)
GBC_fit = GBC_obj.fit(X_train, y_train)
GBC_opt = GBC_fit.best_estimator_


## IV. Results

### PCA 
n_components=150 is selected for PCA algorithm. The most important features and their corresponding meaning in the first 2 PCs are shown in Fig.  and Fig.. Social status and housing holding strongly affects the result in PC1.  

### K-Means
The distributions of population and customers in 10 clusters demonstrates in Fig. and their difference in Fig. Cluster 1,  5, and 7 are overrepresented clusters,  indicating that the highest potential to become customers in the future in those group. 

### Model Evaluation and Validation
The roc_auc scores of four classifiers with default parameters are shown in Fig.  Base on the performance on validation dataset and overfitting situation (Fig. ),  GradientBoostingClassifier is the best model in this case with 0.688.  By grid search and cross validation, the best parameters are chosen in GradientBoostingClassifier. (Fig.)

### Justification
The highest score of  the prediction is 0.88403 so far in Kaggle Competition.  According to the other works,  the performance of "GradientBoostingClassifier" can be around 0.79. But my best score is only 0.736  in the validation dataset. It could result in non-suitable data cleanning. 


### Improvement
-I think that data preprocessing is the key of this project. Extracting inforation effectively affects both the performance of unsupervised and supervised models. My method of  data preprocessing is definitely needs to improve.  For exemple, filling NaNs with mean values is not the best move.  I need to explore more the Attributes spreadsheets to understand better features, so that I can make better decision to keep or drop columns. 

-REPONSE in `Udacity_MAILOUT_052018_TRAIN.csv`only has 1.2% of customers, which will imbalance the model training.

-Improvement in analysis of results.

-----------
