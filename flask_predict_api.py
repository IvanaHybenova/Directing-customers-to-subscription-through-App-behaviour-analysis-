"""
Created on Fri Mar 16 21:06:35 2018

@author: Ivana Hybenoa
"""

import pickle
from flask import Flask, request, make_response, send_file
from flasgger import Swagger
import numpy as np
import pandas as pd
import zipfile
import time
from io import BytesIO


with open('./model/final_model.pkl', 'rb') as model_file:
     model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict_file', methods=["POST"])
def predict_file():
    """Example file endpoint returning a prediction
    ---
    parameters:
      - name: file_input_test
        in: formData
        type: file
        required: true
    responses:
        200:
            description: OK
    """
    dataset = pd.read_csv(request.files.get("file_input_test"))
    
    # Data preprocessing
    
    # Load Top Screens
    top_screens = pd.read_csv('top_screens.csv').top_screens.values
    top_screens
    
    # Mapping Screens to Fields
    dataset["screen_list"] = dataset.screen_list.astype(str) + ','
    
    for sc in top_screens:
        dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
        dataset['screen_list'] = dataset.screen_list.str.replace(sc+",", "")
    
    dataset['Other'] = dataset.screen_list.str.count(",")
    dataset = dataset.drop(columns=['screen_list'])
    
    # Funnels - group of screens that belong to the same set
    # in order to get rid of correlation among some screens and still keep the information
    savings_screens = ["Saving1",
                        "Saving2",
                        "Saving2Amount",
                        "Saving4",
                        "Saving5",
                        "Saving6",
                        "Saving7",
                        "Saving8",
                        "Saving9",
                        "Saving10"]
    # Creating a column with number of visited screens from this funnel  
    dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
    # Dropping the columns from this funnel
    dataset = dataset.drop(columns=savings_screens)
    
    cm_screens = ["Credit1",
                   "Credit2",
                   "Credit3",
                   "Credit3Container",
                   "Credit3Dashboard"]
    # Creating a column with number of visited screens from this funnel
    dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
    # Dropping the columns from this funnel
    dataset = dataset.drop(columns=cm_screens)
    
    cc_screens = ["CC1",
                    "CC1Category",
                    "CC3"]
    # Creating a column with number of visited screens from this funnel
    dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
    # Dropping the columns from this funnel
    dataset = dataset.drop(columns=cc_screens)
    
    loan_screens = ["Loan",
                   "Loan2",
                   "Loan3",
                   "Loan4"]
    # Creating a column with number of visited screens from this funnel
    dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
    # Dropping the columns from this funnel
    dataset = dataset.drop(columns=loan_screens)
    dataset_user = dataset['user']
    dataset = dataset.drop(columns='user')
	
    prediction = model.predict(dataset)
    prediction_probability = model.predict_proba(dataset)
    prediction_probability = prediction_probability[:,1]
    dataset['predicted_class'] = pd.DataFrame(prediction)
    dataset['probability'] = pd.DataFrame(prediction_probability)
    dataset['user'] = pd.DataFrame(dataset_user)
    data = dataset[['user', 'probability', 'predicted_class']]
   
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name='predictions', 
                        encoding='utf-8', index=False)
    writer.save()
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        names = ['predictions.xlsx'] # names = ['iris_predictions.xlsx', 'file2']
        files = [output]  # files = [output, output2]
        for i in range(len(files)):
            input_data = zipfile.ZipInfo(names[i])
            input_data.date_time = time.localtime(time.time())[:6]
            input_data_compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(input_data, files[i].getvalue())
    memory_file.seek(0)
    
    response = make_response(send_file(memory_file, attachment_filename = 'predictions.zip',
                                       as_attachment=True))
    response.headers['Content-Disposition'] = 'attachment;filename=predictions.zip'
    response.headers['Access-Control-Allow-Origin'] = '*'
    
    
    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    