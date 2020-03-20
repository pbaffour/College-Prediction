{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing relevant libraries\n",
    "\n",
    "import numpy as np\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\13023\\Anaconda3\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3334: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "app = Flask(__name__)\n",
    "model = pickle.load(open('model.pkl', 'rb'))\n",
    "\n",
    "# Endpoint: home page\n",
    "@app.route('/',methods=['POST'])\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "# Endpoint: prediction\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    \n",
    "    input_features = [int(x) for x in request.form.values()]\n",
    "    \n",
    "    tmp = input_features[2]\n",
    "    input_features[2] = input_features[1]\n",
    "    input_features[1] = tmp\n",
    "    \n",
    "    final_input = [np.array(input_features)]\n",
    "\n",
    "    prediction = model.predict_proba(final_features)\n",
    "\n",
    "    output = round(prediction[0], 2) * 100\n",
    "\n",
    "\n",
    "\n",
    "    return render_template('index.html', \n",
    "                           prediction_text='Your child has a {}% chance of entering and completing college'.format(output))\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
