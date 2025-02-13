import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class DataCleanerFeatureEngineer:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.price_df = None
        self.label_encoders = {}
    
    def load_data(self):
        """讀取數據"""
        self.price_df = pd.read_csv(self.input_file, dtype={20: str, 21: str})
    
    def process_floor_ratio(self):
        """計算樓高比"""
        self.price_df['樓高比'] = (self.price_df['移轉層次'] / self.price_df['總樓層數']).astype('float64')
        self.price_df['樓高比'] = self.price_df['樓高比'].where(self.price_df['樓高比'] <= 1, 1)
        self.price_df['樓高比'] = self.price_df['樓高比'].fillna(0)
    
    def clean_data(self):
        """移除不必要的欄位並處理房屋年齡"""
        drop_columns = ['編號', '主要建材', '經度', '緯度', '房屋地址', '交易月份', '交易日期', '建築完成年', '建築完成月', '建築完成日', '總價元', '移轉層次', '總樓層數']
        self.price_df.drop(columns=drop_columns, inplace=True)
        self.price_df['房屋年齡'] = self.price_df['房屋年齡'].apply(lambda x: x if x > 0 else None)
        valid_building_types = ['華廈', '住宅大樓', '透天厝', '公寓']
        self.price_df = self.price_df[self.price_df['建物型態'].isin(valid_building_types)]
    
    def encode_categorical(self):
        """對類別變數進行標籤編碼"""
        label_encode_columns = self.price_df.select_dtypes(include=['object']).columns
        for col in label_encode_columns:
            self.price_df[col] = self.price_df[col].fillna('未知')
            encoder = LabelEncoder()
            self.price_df[col] = encoder.fit_transform(self.price_df[col])
            self.label_encoders[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    
    def handle_missing_values(self):
        """填充缺失值為中位數"""
        self.price_df.fillna(self.price_df.median(), inplace=True)
    
    def remove_outliers(self):
        """移除價格異常值"""
        self.price_df = self.price_df[self.price_df["單價元平方公尺"] > 0]
        self.price_df["log_單價元平方公尺"] = np.log1p(self.price_df["單價元平方公尺"])
        self.price_df['log_鄰近區域平均單價'] = np.log1p(self.price_df.groupby('鄉鎮市區')['單價元平方公尺'].transform('mean'))
        threshold = self.price_df["log_單價元平方公尺"].quantile(0.01)
        self.price_df = self.price_df[self.price_df["log_單價元平方公尺"] > threshold]
        self.price_df["高價指標"] = (self.price_df["log_單價元平方公尺"] > 12.5).astype(int)
    
    def save_data(self):
        """儲存清理後的數據"""
        self.price_df.to_csv(self.output_file, encoding='utf-8-sig', index=False)
        print(f"✅ 數據清理與特徵工程完成，儲存至 {self.output_file}")
    
    def process_all(self):
        """執行所有步驟"""
        self.load_data()
        self.process_floor_ratio()
        self.clean_data()
        self.encode_categorical()
        self.handle_missing_values()
        self.remove_outliers()
        self.visualize_data()
        self.save_data()

# ✅ 使用範例
if __name__ == "__main__":
    processor = DataCleanerFeatureEngineer(
        input_file='2014年至2024年歷年實價登錄資料(含生活機能距離).csv',
        output_file='processed_housing_data.csv'
    )
    processor.process_all()