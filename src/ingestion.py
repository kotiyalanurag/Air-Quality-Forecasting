import sqlite3 
import pandas as pd

class Dataset:
    def __init__(self, data_path:str, table_name:str, interpolate:bool, pollutant:str, verbose:bool, save_path:str, dataset:str):
        self.data_path = data_path
        self.conn = sqlite3.connect(self.data_path)
        self.table_name = table_name
        self.interpolate = interpolate
        self.pollutant = pollutant
        self.verbose = verbose
        self.save_path = save_path
        self.dataset = dataset
        self.process_data()

    def process_data(self):
        df = pd.read_sql(f"SELECT * FROM {self.table_name}", self.conn)
        
        df["datetime_obj"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df['date'] = df['datetime_obj'].dt.date
        df['date'] = pd.to_datetime(df['date'])

        daily_avg = df.groupby(["date", "parameter"])["value"].mean().reset_index()

        parameter_data = daily_avg[daily_avg["parameter"] == f"{self.pollutant}"]
        parameter_data = parameter_data.set_index("date").sort_index()

        weekly_data = parameter_data.resample("W")["value"].mean().to_frame()
        weekly_data.rename(columns={"value": f"{self.pollutant}_weekly_avg"}, inplace=True)

        num_missing_weeks = weekly_data[f"{self.pollutant}_weekly_avg"].isna().sum()

        if self.verbose == True:

            print('-'*20 + "Count" + '-'*20)
            print(daily_avg["parameter"].value_counts())
            print('-'*45)
            print('-'*15 + "Data Description" + '-'*15)
            print(weekly_data.describe())
            print('-'*45)
            print('-'*9 + "Number of missing weeks data" + '-'*9)
            print(f"Total: {num_missing_weeks}")
            print('-'*45)
            print('-'*15 + "Data Before" + '-'*15)
            print(f"Shape: {weekly_data.shape}")
            print('-'*45)

        if self.interpolate == True:
            
            weekly_data["target"] = weekly_data[f"{self.pollutant}_weekly_avg"].interpolate(method="linear")
            weekly_data.drop(columns=f"{self.pollutant}_weekly_avg", axis=1, inplace=True)
            
            if self.verbose == True:
                
                print('-'*15 + "Data After Interpolation" + '-'*15)
                print(f"Shape: {weekly_data.shape}")
                print('-'*45)
        
        else:
            
            weekly_data.dropna(inplace=True)
            weekly_data.rename(columns={f"{self.pollutant}_weekly_avg":"target"}, inplace=True)
            
            if self.verbose == True:
                
                print('-'*15 + "Data After Deletion" + '-'*15)
                print(f"Shape: {weekly_data.shape}")
                print('-'*45)

        weekly_data["lag_1"] = weekly_data["target"].shift(1)
        weekly_data["lag_2"] = weekly_data["target"].shift(2)
        weekly_data["roll_3"] = weekly_data["target"].rolling(3).mean()
        weekly_data["roll_6"] = weekly_data["target"].rolling(6).mean()
        weekly_data['baseline_predictions'] = weekly_data['target'].rolling(5).mean()

        weekly_data["target"] = weekly_data["target"].shift(-1)
        weekly_data.dropna(inplace=True)
        weekly_data.to_csv(f"{self.save_path}/{self.dataset}.csv", index=True)

if __name__ == "__main__":
    
    data_path = "./database/raw/openAQ.db"
    table_name = "air_quality"
    interpolate = False
    pollutant = "pm25"
    verbose = True
    dataset = "data"
    save_path = "./dataset"
    
    obj = Dataset(data_path=data_path,
                  table_name=table_name,
                  interpolate=interpolate,
                  pollutant=pollutant,
                  verbose=verbose,
                  dataset=dataset,
                  save_path=save_path)