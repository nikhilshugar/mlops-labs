import tensorflow as tf
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # fixed seed so every build trains the same model
    tf.keras.utils.set_random_seed(42)

    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Build a simple TensorFlow model

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 50 epochs was not enough for the model to finish learning
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.3f}")

    model.save('my_model.keras')
    # save the scaler too, the app needs it to scale inputs the same way
    joblib.dump(sc, 'scaler.pkl')
    print("Model and scaler were trained and saved")