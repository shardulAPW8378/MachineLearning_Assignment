"""
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

"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Step 1: Get Data
data = pd.read_csv("playpredict.csv")

# Step 2: Clean, Prepare and Manipulate data
label_encoder_weather = LabelEncoder()
label_encoder_temperature = LabelEncoder()
label_encoder_play = LabelEncoder()

data['Weather'] = label_encoder_weather.fit_transform(data['Whether'].str.lower())  # Convert to lowercase
data['Temperature'] = label_encoder_temperature.fit_transform(data['Temperature'].str.lower())  # Convert to lowercase
data['Play'] = label_encoder_play.fit_transform(data['Play'])

# Step 3: Train Data
X = data[['Weather', 'Temperature']]
y = data['Play']

# Train the KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Step 4: Test Data
def test_data(weather, temperature):
    weather = label_encoder_weather.transform([weather.lower()])[0]  # Convert to lowercase
    temperature = label_encoder_temperature.transform([temperature.lower()])[0]  # Convert to lowercase
    result = knn.predict([[weather, temperature]])
    return label_encoder_play.inverse_transform(result)  # Inverse transform to get original label

# Suppress warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# Take user input
test_weather = input("Enter the weather (Sunny/Overcast/Rainy): ")
test_temperature = input("Enter the temperature (Hot/Cold/Mild): ")

# Check if input is valid
if test_weather.lower() not in ['sunny', 'overcast', 'rainy'] or test_temperature.lower() not in ['hot', 'cold', 'mild']:
    print("Invalid input! Please enter valid weather (Sunny/Overcast/Rainy) and temperature (Hot/Cold/Mild).")
else:
    print(f"Should we play if weather is {test_weather.capitalize()} and temperature is {test_temperature.capitalize()}?")
    print("Result:", test_data(test_weather, test_temperature))

# Step 5: Calculate Accuracy
def check_accuracy(k_value):
    knn = KNeighborsClassifier(n_neighbors=k_value)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

k_values = [1, 3, 5, 7]  # Different values of K to try
for k in k_values:
    accuracy = check_accuracy(k)
    print(f"Accuracy for k={k}: {accuracy}")


