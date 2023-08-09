import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from datetime import datetime


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), dtype='int32'),
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=128),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    # Read the dataset
    data = pd.read_csv('./字节评论.csv')  # Assuming the data is in 'data.csv'
    data.dropna(subset=['text_column'], inplace=True)  # Drop rows with NaN values

    X = data['text_column']
    y = data['label_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Build and train the model
    model = build_model(input_dim=X_train_vec.shape[1])

    start_time = datetime.now()
    startStr = start_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Training model, start time: {startStr}')

    model.fit(X_train_vec.toarray(), y_train, epochs=10, batch_size=32, verbose=1)

    end_time = datetime.now()
    endStr = end_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Training model, end time: {endStr}')
    print(f'Training model, total time: {(end_time - start_time).total_seconds()} seconds')

    # Evaluate the model
    y_pred = model.predict(X_test_vec.toarray())
    y_pred_binary = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Test Set Accuracy:", accuracy)

    # Save model and vectorizer
    model.save('text_classification_model.h5')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Load model and vectorizer
    loaded_model = tf.keras.models.load_model('text_classification_model.h5')
    loaded_vectorizer = joblib.load('vectorizer.pkl')

    # Use the loaded model for prediction
    while True:
        new_data = input("Enter: ")
        if new_data.lower() == 'exit':
            break
        new_data_vec = loaded_vectorizer.transform([new_data])
        prediction = loaded_model.predict(new_data_vec.toarray())
        print("Prediction:", prediction)


if __name__ == "__main__":
    main()
