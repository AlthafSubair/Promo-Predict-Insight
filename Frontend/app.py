from flask import Flask, render_template, request
import pandas as pd

#created an instance of flask

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    # Displays index page
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)