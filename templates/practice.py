from flask import Flask
import pandas as pd

app = Flask(__name__)

df = None

@app.route('/')
def upload():

    global df

    