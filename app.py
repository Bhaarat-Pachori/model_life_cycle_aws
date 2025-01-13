import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your trained model
with open('nb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# load the vectorizer too
with open('nb_tfidf.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# routes to interact with the model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']  # Assuming your input is a list of features

    new_data_features = loaded_vectorizer.transform(features)

    # for classifying single review we have to reshape the i/p
    new_data_features = new_data_features.reshape(1, -1)

    # Make prediction using the loaded model
    prediction = model.predict(new_data_features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
 