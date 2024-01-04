from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap5

from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

import tensorflow_hub as hub 
import tensorflow as tf
import numpy as np 

import secrets

model = tf.keras.models.load_model(
  "backend/spam_model.h5", 
  custom_objects={'KerasLayer': hub.KerasLayer})

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Bootstrap-Flask requires this line
bootstrap = Bootstrap5(app)
# Flask-WTF requires this line
csrf = CSRFProtect(app)

class NameForm(FlaskForm):
    text = StringField("Text", validators=[DataRequired(), Length(5, 40)])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():

    form = NameForm()
    message = ""
    if form.validate_on_submit():
        text = form.text.data
        prediction = model.predict([text])

        if prediction >= 1:
            return render_template("results.html", result="Spam", prediction=prediction)
        else:
            return render_template("results.html", result="Not Spam", prediction=prediction)
    else:
        message = "Please enter a valid text."

    return render_template("index.html", form=form, message=message)

