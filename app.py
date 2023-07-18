
from flask import Flask , render_template,request,jsonify
from Feuture_extraction.feature_extraction import FeaturExtraction
from Build_Model.build_model import BuilModel
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import numpy as np
from numpy.linalg import norm
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

obj_model_class = BuilModel()
model = obj_model_class.Resnet()

feature_obj = FeaturExtraction()

register_user = pickle.load(open(r'register_feature\register_user_feature.pkl','rb'))
label = ['Anuj','Ranjit','Syam']




app=Flask(__name__)



@app.route('/',methods=['GET','POST'])
def Home():
    return render_template('index.html')
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        uploaaded_file = request.files['file']
        user = int(request.form['user'])

        # to save the file
        basepath = os.path.dirname(__file__)  # to get the bas path
        file_path = os.path.join(basepath, 'uploads', secure_filename(uploaaded_file.filename))   # complete path
        uploaaded_file.save(file_path)

        # to extract the features from image
        features = feature_obj.feature_extractions(file_path,model)
        ## preparing dataset to find the cosine simalarity
        pivot_data = pd.DataFrame({user:register_user[user],'test':features}).T
        simlarity_score = cosine_similarity(pivot_data)

        percentage = round(simlarity_score[0][1]*100,2)





    return render_template('index.html',percentage=f' {percentage}')    # return the value on the webpage
if __name__ == "__main__":
    app.run(debug=True)