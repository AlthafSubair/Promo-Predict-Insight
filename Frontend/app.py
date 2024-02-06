from flask import Flask, render_template, request
import pandas as pd
import sys
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(cur_dir,'..'))

sys.path.append(parent_dir)

from promo_predict_insight import ml_pred


#created an instance of flask

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    # Displays index page
    return render_template('index.html')

# route for handling data submitted by index page form

@app.route('/submit', methods=['POST'])
def submit_form():
    
    # Access form data using request.form
    
    department = request.form.get('department')
    region = request.form.get('region')
    education = request.form.get('education')
    age = request.form.get('age')
    recruitment_channel = request.form.get('recruitment_channel')
    gender = request.form.get('gender')
    no_of_trainings = request.form.get('no_of_trainings')
    length_of_service = request.form.get('length_of_service')
    previous_year_rating = request.form.get('previous_year_rating')
    avg_training_score = request.form.get('avg_training_score')
    awards = request.form.get('awards')
    KPIs_met = request.form.get('KPIs_met')

    # Preprocess the form data to make dataframe 
    
    user_input_df = pd.DataFrame({
        'department': [department],
        'region': [region],
        'education': [education],
        'gender': [gender],
        'recruitment_channel': [recruitment_channel],
        'no_of_trainings': [no_of_trainings],
        'age' : [age],
        'previous_year_rating': [previous_year_rating],
        'length_of_service': [length_of_service],
        'KPIs_met >80%': [KPIs_met],
        'awards_won?': [awards],
        'avg_training_score': [avg_training_score]
    })
    
    #removing coloumn name of dataframe
    
    dis_df = user_input_df.to_numpy()
    
    #pass dataframe to the ML file for predicting result
    
    result = ml_pred(user_input_df)
    
    # Rendering prediction page and passing ML result and dataframe
    return render_template('pred.html',feature=dis_df,pred = result)
    

if __name__=='__main__':
    app.run(debug=True)