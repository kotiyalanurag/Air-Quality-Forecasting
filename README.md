<h1 align=center> Air Quality Forecasting

![](https://img.shields.io/badge/Python-3.14-blue) ![](https://img.shields.io/badge/sklearn-1.7.2-blue) ![](https://img.shields.io/badge/fastapi-0.38.0-blue) ![](https://img.shields.io/badge/docker-7.1.0-blue) ![](https://img.shields.io/badge/LICENSE-MIT-red)</h1>

<p align = left>A system that collects data from OpenAQ API and S3 archives using location coordinates, and predicts weekly air quality levels in terms of NO2, PM2.5, and PM10 parameters through an API offering real-time inference.</p>

## Overview

The model is capable of predicting a weekly AQI value for a region based on observing available previous weeks of AQI readings that go as far as 6 weeks into the past. This particular model achieved a MAE of 3.78 across 52 AQI predictions 
for the city of Aachen located in the North Rhine Westphalia state of Germany for the year 2024. The data for this project
was collected using the OpenAQ api and their S3 bucket archives for past recorded data using AWS.

## Hyperparameters

When we first load the data from our SQLite database the script runs an automatic EDA on the dataset to determine how much missing data we're dealing with. Based on that information we can either drop missing weeks or interpolate those entries from the rest of our dataset. In this case, I've decided to drop those entries since around 40% of the days were missing. 

The hyperparameters for our Gradient Boosting Regressor model are given below:

```python
n_estimators = 1000,       
learning_rate = 0.008,
max_depth = 5,
subsample = 0.8,          
random_state = 42,
validation_fraction = 0.2,  
n_iter_no_change = 50,      
tol = 1e-4
```

## How to run the fastapi app?

Just run the following script from your terminal

```python
$ uvicorn app.app:app
```
Go to the link displayed in terminal and open the swagger UI of fastapi.

## How to run the docker container?

Just build the container using the following script on terminal. Replace my-app-name with a name that you'd like for your image.

```python
$ docker build -t my-app-name .
```
And once the image is built just run a container using

```python
$ docker run -p 8000:8000 my-app-name
```
Just open the link and you'll find that the app is running inside a docker container.

## Results

Here are the results we obtained from 52 weekly prediction in 2024 for the city of Aachen.

<p align="center">
  <img src = figures/XGBresults.png max-width = 100% height = '250' />
</p>

The training scripts for pulling data from the API, cleaning it, and training/fine-tuning our model are given inside the src directory.
