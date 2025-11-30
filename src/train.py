import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

class Boosting:
    def __init__(self, data_path:str, fig_path:str):
        self.data_path = data_path
        self.fig_path = fig_path
        self.train()

    def create_data_splits(self):
        data = pd.read_csv(self.data_path, index_col='date', parse_dates=True)
        train = data.loc[: "2023-12-31"]
        test = data.loc["2024-01-01" :]
        X_train = train.drop(columns=["target", "baseline_predictions"])
        y_train = train["target"]
        X_test = test.drop(columns=["target", "baseline_predictions"])
        y_test = test["target"]

        y_test_baseline = test["target"]
        y_pred_baseline = test["baseline_predictions"]

        return X_train, X_test, y_train, y_test, y_test_baseline, y_pred_baseline 

    def train(self):
        
        X_train, X_test, y_train, y_test, y_test_baseline, y_pred_baseline = self.create_data_splits()
    
        model = GradientBoostingRegressor(n_estimators=1000,       
            learning_rate=0.008,
            max_depth=5,
            subsample=0.8,          
            random_state=42,
            validation_fraction=0.2,  
            n_iter_no_change=50,      
            tol=1e-4)                 

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        preds = np.clip(preds, a_min=0, a_max=None)

        baseline_mae = mean_absolute_error(y_test_baseline, y_pred_baseline)
        print("4 week rolling baseline MAE:", baseline_mae)

        self.plot_results(label=y_test_baseline, prediction=y_pred_baseline, pollutant="no2", figname="baselineresults")

        model_mae = mean_absolute_error(y_test, preds)
        print("XGB MAE:", model_mae)
        self.plot_results(label=y_test, prediction=preds, pollutant="pm25", figname="XGBresults")

        with open("./model/boosting.pkl", "wb") as f:
            pickle.dump(model, f)

    def plot_results(self, label, prediction, pollutant, figname):
        plt.figure(figsize=(14, 6))

        plt.plot(label.index, label, label=f"Actual {pollutant.upper()} Level", linewidth=2.5)
        plt.plot(label.index, prediction, label=f"Predicted {pollutant.upper()} Level", linewidth=2, linestyle="--")

        plt.xlabel("Week", fontsize=14)
        plt.ylabel(f"{pollutant.upper()} Concentration (µg/m³)", fontsize=14)
        plt.title(f"Weekly {pollutant.upper()} Levels: Actual vs Predicted", fontsize=18, weight="bold")
        plt.legend(framealpha=0.9, fontsize=14, shadow=True, fancybox=True)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.fig_path}{figname}.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    data_path = "./dataset/data.csv"
    fig_path = "./figures/"
    obj = Boosting(data_path=data_path,
                   fig_path=fig_path)