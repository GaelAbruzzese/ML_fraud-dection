# Deploy ML API serving predictions
Here you'll be able to deploy a minimal API to return predictions from pretrained model.
     
## Prerequisite
In the following, we suppose that:
 
1. You have previously trained a pipeline
2. You have saved this model either on GCP Cloud Storage or inside `data/` 
3. If your pipeline includes custom transformers, they should be present inside  `TaxiFaremodel/encoders.py`  
    🚨 This point is very important here, please call TA if unclear 🚨
4. You have installed [heroku cli](https://devcenter.heroku.com/articles/heroku-cli)
5. you have created an [free heroku account](https://signup.heroku.com/)

## Summary

Here you will deploy an api that will:  
- load a `model.joblib` using `pipeline = joblib.load('model.joblib')`  
  👉 either locally or directly from GCP Storage  
  👉 this `model.joblib` contains the whole pipeline (preprocssing + model)
- receive through route `/predict_fare`, jsons looking like: 
```python
input = {"pickup_datetime": 2012-12-03 13:10:00 UTC,
        "pickup_latitude": 40.747,
        "pickup_longitude": -73.989,
        "dropoff_latitude": 40.802,
        "dropoff_longitude": -73.956,
        "passenger_count": 2}
```
- apply predictions
- return predictions

# First local API
Before everything let us install the requirements provided inside `requirements.txt`
```bash
pip install -r requirements.txt
```

We will use `flask` python package to develop our API

Inspect `app.py` and check the code of two routes `/predict_fare` and `/` 

For simplicity purpose you will develop your first api locally with a local `model.joblib` stored in `data/` directory that we provide you  
We will see later on how to get your stored model from gcp

Now you can run your api locally by simply running 
```bash
python app.py
```
And check that it work by pinging it: [http://127.0.0.1:8080/](http://127.0.0.1:8080/) 

## Use your api to obtain predictions
Open `Predict.ipynb` under `jupy` folder and start interrogating your api

# Deploy
Now that you checked your app works locally, you might want it to run free on a remote server.  

## Create folder outside data-challenges
Here, as heroku is based on git, we will create a folder outside our gitted `data-challenges`.  
```bash
mkdir -p ~/code/taxifare_api
cp -rf * ~/code/taxifare_api/ 
cd ~/code/taxifare_api/ 
```
Version it for future heroku use:
```bash
cd ~/code/taxifare_api/ && git init
```

## Deploy to heroku
You'll see once again how heroku is easy to use, here you simply need to:
- Create a `Procfile` with following line inside it:
```bash
web: gunicorn app:app
```
```bash
git add . && git commit -am "add api first version"
```

- login to heroku
```bash
heroku login
```

- create app on heroku
```bash
heroku create YOUR_APP_NAME
```

- Build it on heroku servers using git
```bash
git push heroku master
```
- Then deploy
```bash
heroku ps:scale web=1
```
click on the link `https://YOUR_APP_NAME.herokuapp.com/` and it should return `OK`

Update `Predict.ipynb` to request your newly deployed API

# Bonus: Load your own model from GCP
🛑 You can skip this step and come back to it after exercice 2 and/or 3 🛑  

Here we will simply add a function downloading your `model.joblib` from your own storage.   

Inspect new functions under `TaxiFareModel/gcp.py`:
- `download_model()` to get file from storage
- `get_credentials()` to read your json GCP credentials from your env variable `GOOGLE_APPLICATION_CREDENTIALS` 

Replace following variable inside `TaxiFareModel/params.py`
- `BUCKET_NAME`, `PROJECT_ID`, and `MODEL_VERSION`  
  👉 the most important is  `MODEL_VERSION` indicating folder inside your bucket where you previously uploaded your model
  
Test that you correctly get your model from GCP:
```bash
python -m TaxiFareModel.gcp
```
or with `ipython`
```bash
$ ipython
Python 3.7.2 (default, Feb 20 2020, 16:34:30) 
Type 'copyright', 'credits' or 'license' for more information
IPython 7.12.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: %run TaxiFareModel/gcp.py
```

# Bonus: Add route to change model
Inspect already implemented route `/set_model` inside `app.py`:
- What does it do ?
- Test it by launching your app locally
- Deploy this new version on Heroku

⚠️ Beware here, you will need to indicate to heroku your GCP credentials, for that inspect following command from `Makefile`:
```make
heroku_set_gcp_env:
	-@heroku config:set GOOGLE_APPLICATION_CREDENTIALS="$(< /USERS/yourpath/to/yourgcpcredentials.json"
```
and run it right after you redeployed your app
```bash
make deploy_heroku heroku_set_gcp_env 
``` 


