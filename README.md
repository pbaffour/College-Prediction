College Prediction Tool:

I built, trained, and deployed a machine learning model predicting the probability of a high school student 
entering and completing college, based on their mother's highest level of education and reported level of
financial motivation for studying.

I used real, sample data from the National Center for Education Statistics's Education Longitudinal Study of 2002.
Since I used machine learning algorithms, I did not apply weights to have the data be nationally representative.

Overview of files:

model.py - Code for building the machine learning model. Train and test accuracy of 65% and 66%, respectively.

app.py - Flask API for receiving API calls and computing predicted probability based on user inputs.

request.py - Requests module to call Flask APIs.

HTML/CSS - HTML/CSS template for styling page for user to enter survey data about mother's education and student beliefs.
