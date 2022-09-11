# Deep Learning - Predicting Successful Startups

This Jupyter Notebook uses a deep learning approach to predict whether applicants to Alphabet Soup will be successful with their proposed business ventures. A dataset of more than 34,000 organizations is used to build the neural network model. 

Google Colab was used in the development of this Jupyter Notebook. 

## Technologies

This program is written in Python (3.7.13) and developed using Google Colab on a Windows computer. Additional libraries that are used in this application are pandas (1.3.5), matplotlib (3.2.2), scikitlearn (1.0.2), numpy (1.21.6), and tensorflow (2.8.2) (see parenthesis for versions used in program development).

In your Google Colab environment, these lines of code should be sufficient for your import statements: 

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from google.colab import files
import matplotlib.pyplot as plt
```

## Installation Guide

Downloading the code & associated files using `git clone` from the repository is sufficient to download the Jupyter Notebook, ensure that the associated libaries (see Technologies section) are installed on your machine (or in the Google Colab environment) as well. If there are any issues with the library functions please refer to the versions used for app development (see Technnologies section for this information as well).  Please note that this is a Jupyter notebook. 

The code in the notebook is written for the Google Colab environment, there are a few lines of code that you may want to take out if running it on your local machine, such as the lines for file uploads (`from google.colab import files` and `uploaded = files.upload()`). 

## Usage

This notebook is referencing data stored in the Resources folder in the repository, `applicants_data.csv`. As described above, including these lines of code in your notebook will prompt you to add the file to your Google Colab environment: 

```python
from google.colab import files
uploaded = files.upload()
```

If you're not using Google Colab, in the `pd.read_csv` functions, make sure to specify the correct path to the files.

## Code examples

After uploading the data, one hot encoding is used to transform the categorical data into numerical data that the neural network can understand. Simply create a list of categorical variables in your dataframe, transform the data with `OneHotEncoder()`, and concatenate the new dataset with the target and numerical variables from your original dataframe.

```python
categorical_variables_list = ["col1", "col2"]
enc = OneHotEncoder(sparse=False)
encoded_data = enc.fit_transform(df[categorical_variables_list])
encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names(categorical_variables_list)
)
encoded_df = pd.concat([encoded_df, df[["num_col1", "num_col2", "y"]]], axis=1)
```

Prior to running the neural network model on the dataset, the dataset is split into training and testing datasets with `train_test_split()`, and the features data is scaled with `StandardScaler()`.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

Neural network models are then created with various parameters. In this notebook, the neural network models are created with two hidden layers. Models are copmared using different amounts of neurons for each of the layers. Additionally, the features dataset is examined in further detail, and an attempt at simplication of the input dataset is made, with the hope that the neural network model will be optimized as a result. 

Creating a neural network model with 2 hidden layers, compiling the model, and running the model can be done with only a few lines of code: 

python```
nn = Sequential()
nn.add(Dense(units=hidden_nodes_layer1, activation="relu", input_dim=number_input_features))
nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))
nn.add(Dense(units=1, activation="sigmoid"))
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model = nn.fit(X_train_scaled, y_train, epochs=50, verbose=1)
```

Evaluating the model loss & accuracy metrics can be easily done after: 
python```
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
```

Prior to making any additional changes to the dataset in the feature engineering step, the dataset was further explored with evaluating the number of successful organizations based on the data from the categorical columns, simply by grouping by the feature and calculating the percent of organizations with that feature that were successful or not. This step was repeatedly done throughout the dataframe with one line of code: 

python```
df.groupby("categorical_column_name").mean()
```

As a result of this analysis, the following changes to the input dataset were made:
1) The 'STATUS' and 'SPECIAL_CONSIDERATIONS' columns were dropped, as it was determined that they weren't significant predictors (at least with the 2D approach) of whether or not the organization ultimately came out to be successful or not.
2) The 'INCOME_AMT' column was changed from a categorical column to a numerical column, with the thought that the neural network might more easily understand the data in a singular column instead of the form taken as a result of the one hot encoder function.
3) The 'CLASSIFICATION_AMT' column was changed into a numerical column, where the numbers were based on the success rates from the different 'CLASSIFICATION' labels for the most significant extremities (such as the classifications that tended to have companies with success rates >80% or < 20%).  

Afer these "feature engineering" steps, 2 more neural network models were compiled that had a similar structure to the models made before so that the results of this "feature engineering" could more easily be evaluated. 

## Model results

Four different sequential neural network models were evaluated. The models have a lot in common, such as: the number of hidden layers (2 in each), the activation functions (relu on the first two layers and then sigmoid for the output layer), the loss function (binary_crossentropy), the optimizer (adam), the performance metrics (accuracy), and the number of epochs (50).

The first model (model) was made with 5 neurons in each of the hidden layers. The second model (model_A1) was made with 59 and 30 neurons in the first and second hidden layers, respectively. After the "feature engineering" step was completed, neural networks with the same structure as described above were used to evaluate the data, resulting in models 3 (model_2) and 4 (model_2a). 

Believe it or not... the accuracy of each of the neural networks ultimately came out to be 73%. Either there is an error in the code, or none of the optimization attempts actually ended up "optimzing" the neural network model. This Jupyter Notebook may be revisited in the future to improve the accuracy of the overall neural network model.

## References

For more information about the libraries, check out the documentation!

[Tensorflow library](https://www.tensorflow.org/resources/libraries-extensions)

[Pandas library](https://pandas.pydata.org/)

[Matplotlib library](https://matplotlib.org/)

[Numpy library](https://numpy.org/)

[Scikitlearn library](https://scikit-learn.org/stable/)

## Contributors

Project contributors are the Rice FinTech bootcamp program team (instructor Eric Cadena) who developed the tasks for this project along with myself (Paula K) who's written the code in the workbook.
