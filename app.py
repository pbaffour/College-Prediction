# Importing relevant libraries

import numpy as np
from flask import Flask, request, jsonify, render_template
from gevent.pywsgi import WSGIServer
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Endpoint: home page
@app.route('/',methods=['POST'])
def home():
    return render_template('index.html')

# Endpoint: prediction
@app.route('/predict',methods=['POST'])
def predict():
    
    input_features = [int(x) for x in request.form.values()]
    
    tmp = input_features[2]
    input_features[2] = input_features[1]
    input_features[1] = tmp
    
    final_input = [np.array(input_features)]

    prediction = model.predict_proba(final_features)

    output = round(prediction[0], 2) * 100



    return render_template('index.html', 
                           prediction_text='Your child has a {}% chance of entering and completing college'.format(output))


if __name__ == "__main__":
    # Debug/Development
    app.run(debug=True, host="127.0.0.1", port="5000")
    # Production
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()