import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class HousePriceModel:
    def __init__(self, data_path='2014年至2024年歷年實價登錄資料(含生活機能距離).csv', model_path='xgb_model.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test, self.sample_weights = self.load_data()
    
    def load_data(self):
        price_df = pd.read_csv(self.data_path, dtype={20: str, 21: str})
        X = price_df.drop(columns=['單價元平方公尺', 'log_單價元平方公尺'])
        y = price_df['log_單價元平方公尺']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sample_weights = 1 + price_df.loc[X_train.index, "高價指標"] * 1.5
        return X_train, X_test, y_train, y_test, sample_weights
    
    def train_model(self):
        parameters = {
            "objective": 'reg:squarederror',
            "n_estimators": 1000,
            "learning_rate": 0.3,
            "lambda": 1,
            "gamma": 0,
            "max_depth": 5,
            "min_child_weight": 10,
            "verbosity": 1,
            "random_state": 42,
        }
        
        self.model = xgb.XGBRegressor(**parameters)
        self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
        joblib.dump(self.model, self.model_path)
        print(f"模型已成功儲存為 {self.model_path}")
    
    def evaluate_model(self):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        
        y_pred_train = self.model.predict(self.X_train)
        y_pred = self.model.predict(self.X_test)
        rmse_train = mean_squared_error(self.y_train, y_pred_train, squared=False)
        rmse_val = mean_squared_error(self.y_test, y_pred, squared=False)
        print(f"RMSE 訓練集: {rmse_train:.3f} | RMSE 測試集: {rmse_val:.3f}")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].scatter(y_pred_train, self.y_train, alpha=0.5, s=5)
        axes[0].set_xlabel('Predicted values')
        axes[0].set_ylabel('True values')
        axes[0].set_title(f"Training, RMSE: {rmse_train:.3f}")
        
        axes[1].scatter(y_pred, self.y_test, alpha=0.5, s=5)
        axes[1].set_xlabel('Predicted values')
        axes[1].set_ylabel('True values')
        axes[1].set_title(f"Validation, RMSE: {rmse_val:.3f}")
        plt.show()
    
    def predict(self, input_data):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        
        feature_order = self.model.feature_names_in_
        input_df = pd.DataFrame([input_data])[feature_order]
        predicted_price_log = self.model.predict(input_df)[0]
        predicted_price = np.exp(predicted_price_log)
        predicted_total_price = predicted_price * input_data['交易面積']
        return predicted_total_price
    
if __name__ == "__main__":
    model = HousePriceModel()
    model.train_model()
    model.evaluate_model()