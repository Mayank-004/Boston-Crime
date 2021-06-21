
 
# 1. Introduction

 Crime has become an important thing that we need to consider to better our living environment. Especially during the Covid-19 time, many crime scenes are happening, which is dangerous and unacceptable. Social security and personal security have always been the primary concern of our lives.
The project aims to design a powerful interactive web application to detect and detect crime in Boston from 2015 to 2021. Based on historical crime statistics, we can understand what happened in the past year and where the safest place to live. The analysis is getting the information to someone who needs it, such as the police, residents, or investors who want to invest in many aspects of Boston areas such as real estate, retail, and other businesses.
 
 
 # 2. Problem Definition
     
 
 
 From the provided datasets. We will analyze and explore dataset with the help of some of the following questions:
 - Crime pattern in Boston.
- Whether it is possible to predict crime with the help of this dataset.
- Which Area of Boston needs more resources in order to prevent crime?
- Which area has highest number of crime scene?
- Identify trends and relationships between crime types, locations, and their occurrence.
 
 # 3. Methodology and Analysis road map
 
 First, we perform the Exploratory Data Analysis and Data Visualization with Tableau and Python. These tools will help us overview the interesting trend or patterns that occur from the dataset. Then we create five different predictive models to predict crime in Boston areas based on different target features. We make sure to optimize them for the best results. We also apply clustering on some features such as Location, Offense_Code, etc., to see how the crime performs
 in the areas. In the end, we summarize our findings and answer the questions above.


# 4. Dataset

 We are using dataset of the Boston crime incident reporting which is provided by Boston Police Department. This dataset contains the record of crime incidents starting from 2015 till 2021. Data is available on public platform and can be retrieved from their website: Crime Incident Report August 2015 to date. There are seven files, each represent a corresponding year. It is available in “.CSV” file format to view.
Following are the column fields that this dataset contains:
    Column
INCIDENT_NUMBER OFFENSE_CODE OFFENSE_CODE_GROUP OFFENSE_DESCRIPTION DISTRICT
SHOOTING OCCURRED_ON_DATE
Description
Internal BPD report number
Numerical code of offense description
Internal categorization of OFFENSE_DESCRIPTION Primary descriptor of incident
What district the crime was reported in
Indicated a shooting took place.
Earliest date and time the incident could have taken place
             REPORTING_AREA
   RA number associated with the where the crime was reported from.
     

  YEAR
MONTH DAY_OF_WEEK HOUR UCR_PART STREET
Lat
Long
Location
Year component of OCCURRED_ON_DATE
Month component of OCCURRED_ON_DATE
Day of Week component of OCCURRED_ON_DATE Hour component of OCCURRED_ON_DATE Universal Crime Reporting Part number (1, 2, 3) Street name the incident took place
Latitude where the incident took place
Longitude where the incident took place
Latitude and Longitude where the incident took place
                   
 First, we loaded 7 datasets which are crime records from 2015 to 2021, then combined them into one dataset. After checking the missing values, the "UCR_Part" contains many missing values from 2019 to 2021, so we map the values from "OFFENSE_DESCRIPTION" (2015 - 2018) then connect them to the missing values of "UCR_PART". The code is as below:

The dataset contains 465,074 records and 17 columns as below:
We check the overall information of the dataset as below. We can see that the top offense group is Motor Vehicle Accident Response from Investigate person, most of the crime on Friday. We can see more details in the following parts.
Now, we check the numeric variables summary as below:


Data preparation

For this step, we will format the date-time for the variable “OCCURRED_ON_DATE”, fill Null values for Shooting with the value 0. Then I converted “DAY_OF_WEEK” to an ordered category called 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'.

After renamed the columns and change the district codes to district name as below:

After this step, to tidy up the dataset, we defined the Latitude and Longitude, then drop the Lat, Long, and Location in the original dataset.

We also defined a function to set if there is night time after 8PM and day time after 5AM, night time will consider as value 1, day time will be as value 0.
Finally, we have the dataset as below for the EDA part:
 
From the dataset above, we still see UCR_PART still have some missing values, but we will keep that for now because it contains 2019 – 2021 values, because if we remove them, the EDA results will not have the information from 2019 to 2021.


# 5. Exploratory Data Analysis and Data Visualization

First of all, We plotted a correlation plot among the features of dataset to check if there is any multicollinearity exist among the columns. YEAR and OCURRED_ON_DATE columns shows high correlation which is obvious and needs a further improvement.

 We still see UCR_PART still have some missing values from the dataset above, but we will keep that for now because it contains 2019 – 2021 values. Because if we remove them, the EDA

results will not have the information from 2019 to 2021.
Link of the dashboard: Dashboard of Boston Crime Analysis from 2015 to 2021


From the graph above, we can see Crime in Boston reached the highest records from 2016 to 2018 for more than 29,000 records/year, then decreased significantly in 2019 to 66,662 records, and increased again in 2020, and the trend seems to drop in 2021.
Total Boston crime records from 2015 to 2021
What is most occurring crime?
  
The  graphs help us visualize that Motor vehicle accident's most occurring crimes got the highest number of records with 145,567 from 2015 to 2021. Following are larceny, investigate person, drug violence, etc.
When Do Crime occurs?

  Most crimes occur between 12 pm to 8 pm. It is the time which is usually considered an office travel hour and this trend maybe because there is more person on the road at this time. The


crime rate is low between 1 am, and 8 am and increases gradually during the day.
 We can see the crime has slightly differences on weekdays, and the highest crime rate on Fridays and the lowest on Sundays.
 This graph shows July, Aug, and Sep are the months when most crimes occur. The winter months have the lowest crime rates between February and April, and the summer/autumn months have the highest crime rates in June - October.
These are relatively warmer months, and people enjoy their time, usually outside their homes. It may be the reason for the increase in the crime rate. Are there other factors involved in the crime? According to some reports, many types of crime increase during the holidays, especially in robbery and theft. This can happen for many reasons, such as travellers leaving their homes after being robbed. The rise of alcohol and drug leads to the increasing likelihood of a conflict- related crime.
Where Crime occurs?

 graph represent street wise crime. This shows majority of the crime occurred on Huntington Ave, followed by River St and Warren St.

Data shows Roxbury is highest crime area, followed by South end and Dorchester.
Following with the highest crime report by districts, in Dorchester, we can see that in Mattapan, at Blue Hill Ave street shows the highest crime record. The following street is Dorchester Ave which also a dangerous area.



Correlation Matrix

With the help of a correlation matrix, we can visualize there is no multicollinearity in our dataset. Occurred on date and Year seems highly related, which as usual, we have taken care of this relationship. From this result, it will be helpful for further analysis in the next step.


# 6. Data cleaning

First, we need to check the missing values of the dataset. We can notice that variable Group contains more than 24% missing values of the dataset. Following is UCR_Part, Street and District.
  For further analysis, we remove columns with missing values that have more than 10%, then with UCR_Part, we fill Null values with “Other”, then drop the rows with no values.


# 7. Prediction Analysis – Model building

Different models were applied to choose the final models with the best performance to help predict crime in Boston. The models we used are K-nearest neighbors’ algorithm (KNN), Random Forest, Light Gradient Boosting Machine (LightGBM), Gaussian Naive Bayes (GaussianNB), and Artificial neural networks (ANNs). First, we defined a reusable function to evaluate model performance:
 
Then, we removed the unwanted features and keeping the remains features that we need to use for the predictive models. Our target variable is UCR_PART.
After that, we moved to the encoding step when we take in the dataset. Returns label encoded data frame and the label encoder, models. Then, we split the relevant variables into matrices and

  
split them into train and test groups with a ratio of the test size is 0.33.
19
 K-nearest neighbors’ algorithm (KNN) Model
For this model, we set p = 2 as equivalent to using the Euclidean distance. Weights = 'distance' means closer points are weighted more heavily than further away points, and the K neighbor is 5. After training the model and predict with the test set. We have the result as below:
The accuracy got 43% for the F1-score.
 
Random Forest

We use Random Forest model because random forest builds multiple decision trees and merges them to get a more accurate and stable prediction (Niklas Donges, 2019). For this model, we set the estimators = 100, max_depth=20. We have the result as below:
The accuracy got 49% for the F1-score.
Light Gradient Boosting Machine (LightGBM)
We use LightGBM because it extends the gradient boosting algorithm by adding a type of automatic feature selection and focusing on boosting examples with larger gradients. This can result in a dramatic speedup of training and improved predictive performance (Jason Brownlee, 2020). After training and testing the model, we got the result as below:

The accuracy got 49.7% for the F1-score.

Gaussian naive Bayes (GaussianNB)

A Gaussian Naive Bayes algorithm is a special type of NB algorithm. It's specifically used when the features have continuous values. It's also assumed that all the features are following a gaussian distribution i.e, normal distribution (Rahul Saxena, 2017). We have the result as below:

 The accuracy got 46% for the F1-score.

Artificial Neural Network (ANNs)

ANNs can identify and learn correlative patterns between input and output data pairs. Once trained, they may be used to predict outputs from new sets of data. One of the most useful properties of ANNs is their ability to generalize (Yang, Z. Yang, 2014). We have the result as below:
 As the accuracy got 50% for the F1-score. This model got the highest performance in overall.

Discussion

The prediction accuracy is about 50%, so it is clear that the results are not perfect. One possible explanation for this low productivity is the complexity of crime distribution. In other words, information about location and time is not enough to make a perfect model. We might need additional information to create a more accurate model, such as the information about the victims like gender, age, etc. Because data is complex for future work, one possible approach is to use advanced models to get a better result and avoid limitations on data information.
Conclusion
After the analysis of the Boston crime dataset, we can clearly see the trends and relationships between crime type, location, and incidence. Here are some notable findings from the analysis:
- The most dangerous districts in Boston are Dorchester, South-End, Roxbury.
- Most of the crimes were recorded in the summer months of July and August.
- The Boston Police Department reported the highest number of reactions to motor
vehicle accidents.
 This analysis can help Boston Police act accordingly and reduce the crimes that are
common in the City of Boston.
 Frequency of District Generated Events: Dorchester, Roxbury, and South End District have
the most cumulative events in the entire data set. The three cities are also neighbors, stretching from the south to the southeast of Boston.
For overall process, first, we have reviewed the theoretical concepts of modeling methods to understand the dilemmas applied in the dataset. Second, we visualized the dataset from different angles, this process allowed us to better understand the data and create ideas about how to structure the data. In addition, we created a model based on the information obtained and used the model to predict the test data set. Finally, we compared the results and offered some possible explanations for the outcome.
On the modeling side, we applied 5 models. Theoretically, these models are suitable for classification problems. From the results, we can see that the Artificial Neural Network is a little better; However, this minor correction may take time. This work presents an initial reference for

future applications in the data analysis of criminal statistics according to geographical and temporal characteristics and extends to other datasets in the fields of public order, business and law.


# References,


Crime Incident Eeports (August 2015 - to date) (source: new system), Analyze Boston, from
https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system

A complete guide to the random forest algorithm, Niklas Donges, 2019, from
https://builtin.com/data-science/random-forest-algorithm

How to Develop a Light Gradient Boosted Machine (LightGBM) Ensemble, Jason Brownlee, 2020, from https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm- ensemble/

Gaussian naive bayes classifier implementation in python, (Rahul Saxena, 2017), from
https://dataaspirant.com/gaussian-naive-bayes-classifier-implementation-python/

Artificial Neural Network, Yang, Z. Yang, 2014, from
https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/artificial- neural-network

Machine Learning Algorithms for Visualization and Prediction Modeling of Boston Crime Data, Research Gate, Jiarui YinInikuro, Michael AfaInikuro, John AfaIduabo, 2020, from https://www.researchgate.net/publication/339214659_Machine_Learning_Algorithms_for_Visua lization_and_Prediction_Modeling_of_Boston_Crime_Data
Crimes in Boston | Multiclass / Clustering, Kaggle, retrieved from
https://www.kaggle.com/kosovanolexandr/crimes-in-boston-multiclass-clustering
          
24
