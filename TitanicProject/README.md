# Titanic-ML-PredictiveModel
This repository will store my project which predicts the death of people on the Titanic using the Titanic Dataset from Kaggle.

## Kaggle
The link to the kaggle competition: https://www.kaggle.com/c/titanic/overview

## Starting the Predictive Model
I have added the Data Analyses python file. This file runs some analysis and draws graphs to show which columns should be used for the predictive models.


## Data Analysis
To start off I had to look at the actual training dataset to get a feel of what type of data I would be working with.

#### This shows the First 5 Rows of the Dataset:

![alt-text](img/Train_Head.png)




### Analysis on Gender/Age:

To start off analysing this dataset I needed to decide what variables I would like to look at first. 
The __First__ set of varibales I looked at were Age and Sex
I used the data to create two histograms, one for **Male** and one for **Female**.
They both show Survival rate with the **Age** on the x-Axis.

![Histogram](img/Histogram.png )

Looking at the histograms we can see that:
 * More female passengers survived than male
 * Lower survival rate for male passengers between the age of 15-30 
 * Higher survival rate for female passengers between the ages of 15-40


### Analysis on Survived/Pclass/Gender/Embarked:

The second analysis that I carried out on the dataset was using the **Survived/Pclass/Gender/Embarked** variables. This is comparing how the class of the ticket to the survival rate of both female and male. Whilst also showing this from the three different ports that the Titanic stopped at. 

#### First Graph:

<table>
 <div class="col-md-6" img> 
  <img align='left' src="img/Pclass.png" width="310" height="510" hspace="20"/>   
 <div>
 <div class="col-md-6" text>
  <tr>
   <ul> Looking at the graphs we can see that:
    <li> Females that embarked at ports S and Q had a higher probabilty to survive compare to port C where men had a higher            probability to survive
    <li> Overall the first class ticket holders had a higher chance of survival compared to the other classes 
  </ul>
  </tr>
 </div>
</table>

#### Second Graph:
<table>
 <div calss='col=md-6'> 
  <img align='left' src="img/Survived.png" width="450" height="250" hspace="20"/>
 <div>
 <div class="col-md-6" text>
  <tr>
   <ul> Looking at the graph:
    <li> The most of the passengers that didnt survive came from the 3rd class
    <li> Out of the passengers that survived most were the 1st class and the second most was the 3rd class
   </li>
  </tr>
 </div>
</table>
 
 
 
## Data Preprocessing

One of the main things I had to consider was data preprocessing. The raw data provided for this project was no where near ready to be put into a ML model and used to predict anything just yet. 

### Null Values

My first course of action was to find how many null/NaN values there were in this dataset, I made a new dataframe which had all the null values for the training data for each column:

<img src="img/Missing_value.png" width="300" height="375">

Looking above at the dataframe we can see that **Cabin** had the most null values with 687 which is 77.1% of the total data for that column. **Age** also has a significant value of nulls with 177 which is 19.9% of the total data for that column. The **Cabin** column has nearly 80% of the values missing and the **PassengerId** column will not affect the outcome of the predictions. The **Name** column will not add too much of a significant value to the model as well as the **Ticket** column as it has too many unique values which will skew the accuracy of the predictive model. Therefore I have decided to exclude all 4 of these column from the training and testing data.

#### Age

To deal with the **Age** column which has 177 values missing I decided to calculate the mean and the standard deviation of the age column and then generate random numbers which are focused around those results to fill in the null values. This is done through a simple for loop which uses both the training and test data to fill in the blanks with the numpy library randint function. 

#### Embarked

As there are only 2 values for the **Embarked** column it is best to fill those two values in with the most common port that people embarked on. That port is port "S" which was found through the .describe() function.

#### Sex

The **Age** column has string varibales which are "male" and "female", I have chosen to convert them to 1 and 0 as it will be easier for the models to understand integers.

#### Embarked

I have change the "C", "S", "Q" from the **Embarked** column to 0, 1, 2. 

## Numeric Dataset

The dataset now is fully numeric and has no null values: 

<img src="img/Numeric_Data.png" width="540" height="270">


## Training the ML Models

### Random Forest Model
A Random Forest is as it suggests, it creates a forest and makes the forest all random. This is done through creation of Descion Trees using the bagging method which is a very powerful ensemble method (An ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model)
For this method I will be using the **Sklearn** python library which contains many Machine Learning methods which can be called by th use of one line. 
Inside this code the default setting for **n_estimators**(The number of trees in the forest) is set to 10, however playing around with the number I have concluded that 150 was a good value of trees to use, **min_samples_leaf**(The minimum number of samples required to be at a leaf node) was set to 3, **max_features**(The number of features to consider when looking for the best split) was set to 0.5 and for **n_jobs**(The number of jobs to run in parallel for both fit and predict) was at -1. 

The **Score** for the model was: **0.9181708784596871**

The **Accuracy Score** for the model was: **0.8666666666666667**




