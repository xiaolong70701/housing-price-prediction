import pandas as pd
import numpy as np
import json

class PriceProcessor:
    def __init__(self, price_df):
        """
        初始化 PriceProcessor 類別
        :param price_df: pandas DataFrame，包含 '交易年份', '鄉鎮市區', '單價元平方公尺' 欄位
        """
        self.price_df = price_df
    
    def compute_avg_ln_price(self):
        """
        計算每個 '交易年份' 和 '鄉鎮市區' 的 '單價元平方公尺' 平均值並取自然對數
        :return: pandas DataFrame，包含 '交易年份', '鄉鎮市區', 'ln_單價元平方公尺'
        """
        avg_price_df = self.price_df.groupby(['交易年份', '鄉鎮市區'])['單價元平方公尺'].mean().reset_index()
        avg_price_df['ln_單價元平方公尺'] = np.log(avg_price_df['單價元平方公尺'])
        return avg_price_df
    
    def save_to_json(self, filename):
        """
        計算結果並存成 JSON 檔案
        :param filename: str, JSON 檔案名稱
        """
        avg_price_df = self.compute_avg_ln_price()
        
        # 轉換成 JSON 可接受的格式，使用巢狀字典
        result_json = {}
        for _, row in avg_price_df.iterrows():
            year = str(row['交易年份'])  # 轉成字串
            district = str(row['鄉鎮市區'])  # 轉成字串
            ln_price = row['ln_單價元平方公尺']

            # **處理 NaN 值，轉換為 None（儲存成 JSON 時會變成 null）**
            if pd.isna(ln_price):
                ln_price = None

            if year not in result_json:
                result_json[year] = {}
            result_json[year][district] = ln_price

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        
        print(f"JSON 檔案已儲存為 {filename}")

if __name__ == "__main__":
    price_df = pd.read_csv("/Users/anthonysung/python/Housing/2014年至2024年歷年實價登錄資料(含生活機能距離).csv")
    price_processor = PriceProcessor(price_df)
    price_processor.save_to_json("average_price_per_sqm.json")
