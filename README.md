# Deep_nueral_network36
--

# Social Media Cyber Attacks Analysis
Scenario
The rise of social media platforms in the USA has transformed the way people communicate and share information. With their increasing popularity, these platforms have also become prime targets for cyber-attacks, which can range from data breaches to spreading misinformation. It is essential for companies and individuals to understand the patterns of these attacks to prevent potential threats in the future.
# Objective
To develop a deep learning model using TensorFlow and Keras that can predict the likelihood of a social media account or post being a target of a cyber-attack based on various input features.
# Tasks
- Data Loading & Exploration

- Import the necessary library .

- Load the dataset into a DataFrame using pd.read_csv().

- Display the first few rows of the DataFrame using the head() function to visually inspect its structure.

- Data Preprocessing

- Import the TfidfVectorizer from sklearn.feature_extraction.text.

- Convert the text column 'post_content' into numerical data using the TF-IDF vectorizer.

- Categorical Data Transformation:

- Convert categorical columns like 'platform' and 'account_type' into one-hot encoded columns using pd.get_dummies().
Numerical Data Normalization:

- Import StandardScaler from sklearn.preprocessing.

- Normalize numerical columns like 'followers', 'num_posts', 'likes', etc. using the StandardScaler.

  Data Split:

- Import train_test_split from sklearn.model_selection.

- Split the data into training, validation, and test sets using train_test_split().

- Concatenate all the transformed and normalized features to form the final feature matrix X.

- Assign the target column 'cyber_attack' to y.

Model Building

- Import required modules from tensorflow and keras.

- Define a sequential neural network model with dense layers and dropout layers using keras.Sequential().

- Compile the model using the 'adam' optimizer, 'binary_crossentropy' as the loss function, and track 'accuracy' as a metric.

Model Training

- Use the fit() method of the model to train it on the training data (X_train, y_train).

- Specify a certain number of epochs (iterations over the entire dataset) for the training.

- Use the validation data (X_val, y_val) to validate the model's performance during training. This helps in monitoring overfitting.

Model Evaluation

- Use the evaluate() method of the model to assess its performance on the test data (X_test, y_test).

- This will provide the loss and accuracy of the model on the test data.

Predictions

- Use the predict() method of the model to get the probability scores of the cyber attacks for new data instances.

- Convert the probability scores to binary class labels (e.g., 0 for "no attack" and 1 for "attack") based on a threshold, typically 0.5.

Model Tuning
- Experiment with different architectures for the neural network (e.g., more layers, different activation functions).

- Adjust hyperparameters like the learning rate, batch size, or dropout rate.

- Use techniques like cross-validation or grid search to find the best hyperparameters.

Conclusion and Insights

- Summarize the findings and provide insights from the analysis.

# 1. Data Loading & Preprocessing
``` python
# Import the required library
import pandas as pd  # Import the pandas library and alias it as 'pd' for convenience.

# Import necessary functions/classes from scikit-learn library
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder  # Import specific preprocessing tools from scikit-learn.

# Import the train_test_split function from scikit-learn
from sklearn.model_selection import train_test_split  # Import a function for splitting data into training and testing sets.

# Import TensorFlow and Keras
import tensorflow as tf  # Import TensorFlow, a popular deep learning framework.
from tensorflow import keras  # Import Keras, the high-level neural networks API that runs on top of TensorFlow.

# Load the dataset
data = pd.read_csv('social_media_cyber_attack_dataset.csv')  # Read the CSV file 'social_media_cyber_attack_dataset.csv' into a pandas DataFrame.
data.head()  # Display the first few rows of the DataFrame to inspect the data.
```
Data preprocessing
- TF-IDF, which stands for Term Frequency-Inverse Document Frequency, is a numerical technique used in natural language processing (NLP) to convert text data into numerical form.
- It is particularly useful for representing the importance of words in a document within a larger corpus of documents.
- TfidfVectorizer is used to create a TF-IDF representation of text data with a maximum of 100 features (terms) to capture the most significant information from the documents.
```python
# Import the TfidfVectorizer from scikit-learn for text data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer with a maximum of 100 features
tfidf = TfidfVectorizer(max_features=100)

# Transform the 'post_content' text data into a TF-IDF numerical representation and convert it to a dense array
post_content_transformed = tfidf.fit_transform(data['post_content']).toarray()

# Convert categorical data ('platform' and 'account_type') into one-hot encoding with appropriate prefixes
platform_encoded = pd.get_dummies(data['platform'], prefix='platform')
account_type_encoded = pd.get_dummies(data['account_type'], prefix='account_type')

# Create a StandardScaler object for numerical data normalization
scaler = StandardScaler()

# Normalize numerical features ('followers', 'num_posts', 'likes', 'shares', 'comments', 'account_age') using StandardScaler
scaled_features = scaler.fit_transform(data[['followers', 'num_posts', 'likes', 'shares', 'comments', 'account_age']])

f_scaled_df = pd.DataFrame(scaled_features, columns=['followers', 'num_posts', 'likes', 'shares', 'comments', 'account_age'])

# Concatenate all the processed features (TF-IDF, one-hot encoded categorical, past_history, and scaled numerical)
X = pd.concat([pd.DataFrame(post_content_transformed), platform_encoded, account_type_encoded, data['past_history'], pd.DataFrame(scaled_features)], axis=1)

# Extract the target variable ('cyber_attack')
y = data['cyber_attack']

# Split the dataset into training, validation, and test sets using train_test_split
# First, split into training (70%) and temporary (30%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Then, split the temporary set into validation (15%) and test (15%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)



# Print the shapes of the resulting training, validation, and test sets
X_train.shape, X_val.shape, X_test.shape

pd.DataFrame(post_content_transformed)

platform_encoded

account_type_encoded

X.columns
X.head()
y
X_train.shape[1]

# Define a sequential neural network model
model = keras.Sequential([
    # Add a dense layer with 128 units, ReLU activation function, and input shape matching the number of features
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),

    # Dropout refers to dropping out nodes in a neural network. The connections with a dropped node are temporarily removed, creating a new network architecture.
    # Layers are dropped by a dropout probability of p.
    # Add a dropout layer with a dropout rate of 20% (0.2)
    keras.layers.Dropout(0.2),

    # Add another dense layer with 64 units and ReLU activation
    keras.layers.Dense(64, activation='relu'),

    # Add another dropout layer with a dropout rate of 20% (0.2)
    keras.layers.Dropout(0.2),

    # Add the output layer with 1 unit and sigmoid activation (for binary classification)
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with the Adam optimizer and binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture, displaying layer information and parameter counts
model.summary()

```
- An epoch refers to one complete pass of the entired training dataset through the neural network.

- When all the data samples have been exposed to the neural network for training, one epoch is said to be completed.
- Batch: a set of N sample from the dataset.

- Batch size: number of samples that pass through network before model parameters are updated.

- Iteration: update of model's parameters. Each iteration is the number of batches needed to complete one epoch.

- Example 1: 1000 training samples, batch size=500. It will take 2 iterations to complete 1 epoch.

- Example 2: Dataset with 200 samples. 1000 epochs, or 1000 turns for the datset to pass through the model. Batch size is 5. Then model weights are updated when each of the 40 batches (of size 5) passes through. The model will be updated 40 times.

- The validation set is used during the training phase of the model to provide an unabiased evaluation of the model's performance and tune the model's parameters.
- The test set is used after the model has been fully trained to assess the model's performance on new unseen data.
# 3. Model Training & Evaluation

``` python
# Train the model using the training data
# Fit the model to the training data, validate on the validation data, run for 10 epochs, and use a batch size of 32
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
# validation data - data on which to evaluate the loss and other metrics at the end of each epoch.
#  IMPORTANT: The model will NOT be trained on the validation data.


# Evaluate the model's performance on the test set
# Calculate the loss and accuracy on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the test accuracy
test_accuracy
# Select a subset of sample data (first 10 rows) from the test set for prediction
sample_data = X_test.iloc[:10]

# Select the corresponding true labels for the sample data
sample_labels = y_test.iloc[:10]

# Use the trained model to make predictions on the sample data
predictions = model.predict(sample_data)

# Convert the continuous prediction probabilities into binary labels (0 or 1) using a threshold of 0.5
predicted_labels = [1 if p > 0.5 else 0 for p in predictions]

# Combine the predicted labels and true labels into pairs using zip, then convert them to a list for easier inspection
list(zip(predicted_labels, sample_labels))
```
# 5. Report and Analysis
Based on the results:

1- The model achieved an accuracy of approximately as given above on the test set.
2- The sample predictions show the model's ability to predict the likelihood of cyber attacks on new, unseen data.
3- Further improvements can be made by tuning the model, using more complex architectures, or incorporating additional relevant features.
4- This solution provides a foundational approach to analyzing cyber attacks on social media using deep learning. Real-world implementations might require more sophisticated models and preprocessing techniques.

link to access CSV:

https://drive.google.com/file/d/1AW10bFCZNmTh4OD6nwSOzpC4pRxASyUO/view?usp=drive_link

