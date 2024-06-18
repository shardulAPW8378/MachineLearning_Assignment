# MachineLearning_Assignment
In In this repository i have put all my work that i learned and practiced while learning and working on Machine Learning 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Assignment1:
There is one data set of wether conditions. 
That dataset contains information as wether and we have to decides whether to play or 
not. 
Data set contains the target variable as Play which indicates whether to play or not.
According to above dataset there are two features as 
1.Wether 
2.Temperature 
We have two labels as 
1.Yes 
2.No 
There are three types of different entries under Wether as 
1.Sunny 
2.Overcast 
3.Rainy 
There are three types of different entries under Temperature as 
1.Hot 
2.Cold 
3.Mild 

Design machine learning application which follows below steps as 
Step 1: 
Get Data 
Load data from MarvellousInfosystems_PlayPredictor.csv file into python application. 

Step 2: 
Clean, Prepare and Manipulate data 
As we want to use the above data into machine learning application we have prepare 
that in the format which is accepted by the algorithms. 
As our dataset contains two features as Wether and Temperature. We have to replace 
each string field into numeric constants by using LabelEncoder from processing module 
of sklearn. 

Step 3: 
Train Data 
Now we want to train our data for that we have to select the Machine learning algorithm. 
For that we select K Nearest Neighbour algorithm. 
use fit method for training purpose. For training use whole dataset. 

Step 4: 
Test Data 
After successful training now we can test our trained data by passing some value of 
wether and temperature. 
As we are using KNN algorithm use value of K as 3. 
After providing the values check the result and display on screen. 
Result may be Yes or No. 

Step 5: 
Calculate Accuracy 
Write one function as CheckAccuracy() which calculate the accuracy of our algorithm. 
For calculating the accuracy divide the dataset into two equal parts as Training data and 
Testing data. 
Calculate Accuracy by changing value of K.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Assignment2
There is one dataset of wine which classify the wines according to its contents into three classes 
Marvellous Infosystems: Python-Automation & Machine Learning

These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

Wine data set contains 13 features as
1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins 10)Color intensity
11) Hue
12)OD280/OD315 of diluted wine
13)Proline

According to the above features wine can be classified as
• Class 1
• Class 2
• Class 3

Design machine learning application which follows below steps as

Step 1:
Get Data
Load data from WinePredictor.csv file into python application.

Step 2:
Clean, Prepare and Manipulate data
As we want to use the above data into machine learning application we have prepare that in the format which is accepted by the algorithms.

Step 3:
Step Train Data
Now we want to train our data for that we have to select the Machine learning algorithm.
For that we select K Nearest Neighbour algorithm.
use fit method for training purpose.
For training use 70% dataset and for testing purpose use 30% dataset.

Step 4:
Test Data
After successful training now we can test our trained data by passing some value of
wether and temperature.
As we are using KNN algorithm use value of K as 3. After providing the values check the result and display on screen.
Result may be Yes or No.

Step 5:
Calculate Accuracy
Write one function as CheckAccuracy() which calculate the accuracy of our algorithm. For calculating the accuracy divide the dataset into two equal parts as Training data and Testing data.
Calculate Accuracy by changing value of K.
Before designing the application first consider all features of data set.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
