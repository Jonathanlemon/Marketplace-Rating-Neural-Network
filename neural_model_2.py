import tensorflow as tf
import pandas as pd
import nltk
import sklearn
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the NLTK and SpaCy packages
#nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Load the data from CSV file
data_file = "data2.csv"
df = pd.read_csv(data_file)

# Extract the input and output columns
full_review_col = "full_review"  # Column name for the input full_review
overall_rating_col = "rating"  # Column name for the overall_rating output
seller_rating_col = "seller_rating"  # Column name for the seller_rating output
product_rating_col = "product_rating"  # Column name for the product_rating output
shipping_rating_col = "shipping_rating"  # Column name for the shipping_rating output

# Define the data and labels
data = df[full_review_col].astype(str).values.tolist()
overall_rating_labels = df[overall_rating_col].values.tolist()
seller_rating_labels = df[seller_rating_col].values.tolist()
product_rating_labels = df[product_rating_col].values.tolist()
shipping_rating_labels = df[shipping_rating_col].values.tolist()

# Convert labels to one-hot encoded arrays
overall_rating_labels_onehot = tf.keras.utils.to_categorical(overall_rating_labels, num_classes=6)
seller_rating_labels_onehot = tf.keras.utils.to_categorical(seller_rating_labels, num_classes=6)
product_rating_labels_onehot = tf.keras.utils.to_categorical(product_rating_labels, num_classes=6)
shipping_rating_labels_onehot = tf.keras.utils.to_categorical(shipping_rating_labels, num_classes=6)

# Split data into training and testing sets
data_train, data_test, overall_rating_labels_train, overall_rating_labels_test, \
seller_rating_labels_train, seller_rating_labels_test, \
product_rating_labels_train, product_rating_labels_test, \
shipping_rating_labels_train, shipping_rating_labels_test = train_test_split(data,
                                                                              overall_rating_labels_onehot,
                                                                              seller_rating_labels_onehot,
                                                                              product_rating_labels_onehot,
                                                                              shipping_rating_labels_onehot,
                                                                              test_size=0.1,
                                                                              random_state=42)

data_train, data_val, overall_rating_labels_train, overall_rating_labels_val, \
seller_rating_labels_train, seller_rating_labels_val, \
product_rating_labels_train, product_rating_labels_val, \
shipping_rating_labels_train, shipping_rating_labels_val = train_test_split(data_train,
                                                                              overall_rating_labels_train,
                                                                              seller_rating_labels_train,
                                                                              product_rating_labels_train,
                                                                              shipping_rating_labels_train,
                                                                              test_size=0.2,
                                                                              random_state=42)

# Vectorize the input data using TF-IDF
vectorizer = TfidfVectorizer()
data_train_vec = vectorizer.fit_transform(data_train)
data_test_vec = vectorizer.transform(data_test)
data_val_vec = vectorizer.transform(data_val)
data_train_vec.sort_indices()
data_test_vec.sort_indices()
data_val_vec.sort_indices()
# Define the neural network model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(data_train_vec.shape[1],)))
#model.add(tf.keras.layers.Dense(1028, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(512, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(256, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for overall rating
history = model.fit(data_train_vec, overall_rating_labels_train, epochs=15, batch_size=30, verbose=1, validation_data=(data_val_vec, overall_rating_labels_val))
# Evaluate the model for overall rating
overall_rating_scores = model.evaluate(data_test_vec, overall_rating_labels_test, verbose=1)
print('Overall Rating Loss:', overall_rating_scores[0])
print('Overall Rating Accuracy:', overall_rating_scores[1])

plt.plot(history.history['accuracy'], label='Overall Rating Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 1])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Overall Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 3])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

# Train the model for seller rating
history = model.fit(data_train_vec, seller_rating_labels_train, epochs=15, batch_size=30, verbose=1, validation_data=(data_val_vec, seller_rating_labels_val))
#Evaluate
seller_rating_scores = model.evaluate(data_test_vec, seller_rating_labels_test, verbose=1)
print('Seller Rating Loss:', seller_rating_scores[0])
print('Seller Rating Accuracy:', seller_rating_scores[1])

plt.plot(history.history['accuracy'], label='Seller Rating Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 1])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Seller Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 3])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

# Train the model for seller rating
history = model.fit(data_train_vec, product_rating_labels_train, epochs=15, batch_size=30, verbose=1, validation_data=(data_val_vec, product_rating_labels_val))
#Evaluate
product_rating_scores = model.evaluate(data_test_vec, product_rating_labels_test, verbose=1)
print('Product Rating Loss:', product_rating_scores[0])
print('Product Rating Accuracy:', product_rating_scores[1])

plt.plot(history.history['accuracy'], label='Product Rating Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 1])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Product Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 3])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

# Train the model for seller rating
history = model.fit(data_train_vec, shipping_rating_labels_train, epochs=15, batch_size=30, verbose=1, validation_data=(data_val_vec, shipping_rating_labels_val))
#Evaluate
shipping_rating_scores = model.evaluate(data_test_vec, shipping_rating_labels_test, verbose=1)
print('Shipping Rating Loss:', shipping_rating_scores[0])
print('Shipping Rating Accuracy:', shipping_rating_scores[1])

plt.plot(history.history['accuracy'], label='Shipping Rating Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 1])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='Shipping Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.ylim([0.1, 3])
plt.xlim([0,14])
plt.legend(loc='lower right')
plt.show()