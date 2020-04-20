from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    Cylinders = sample_json['cylinders']
    Displacement = sample_json['displacement']
    Horsepower = sample_json['horsepower']
    Weight = sample_json['weight']
    Acceleration = sample_json['acceleration']
    Year = sample_json['year']
    Origin = sample_json['origin']    
    
    X = [[Cylinders,Displacement,Horsepower,Weight,Acceleration,Year,Origin]]
    
    prediction = model.predict(X)
    
    
    return prediction



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("Pattara_Model.h5")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    Cylinders = TextField('Cylinders')
    Displacement = TextField('Displacement')
    Horsepower = TextField('Horsepower')
    Weight = TextField('Weight')
    Acceleration = TextField('Acceleration')
    Year = TextField('Year')
    Origin = TextField('Origin')


    submit = SubmitField('Analyze')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['Cylinders'] = form.Cylinders.data
        session['Displacement'] = form.Displacement.data
        session['Horsepower'] = form.Horsepower.data
        session['Weight'] = form.Weight.data
        session['Acceleration'] = form.Acceleration.data
        session['Year'] = form.Year.data
        session['Origin'] = form.Origin.data



        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['cylinders'] = int(session['Cylinders'])
    content['displacement'] = float(session['Displacement'])
    content['horsepower'] = float(session['Horsepower'])
    content['weight'] = int(session['Weight'])
    content['acceleration'] = float(session['Acceleration'])
    content['year'] = int(session['Year'])
    content['origin'] = int(session['Origin'])
    
    

    results = return_prediction(model=flower_model,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)
