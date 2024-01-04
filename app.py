from flask import Flask, render_template
from flask_bootstrap import Bootstrap5

from flask_wtf import FlaskForm, CSRFProtect
from wtforms import SubmitField, TextAreaField
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

bootstrap = Bootstrap5(app)

csrf = CSRFProtect(app)

class NameForm(FlaskForm):
    text = TextAreaField("", validators=[DataRequired(), Length(5)])
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

    return render_template("index.html", form=form, message=message)

