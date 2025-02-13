import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

class FacilityProcessor:
    def __init__(self, house_data_path, facility_folder, output_path):
        self.house_data_path = house_data_path
        self.facility_folder = facility_folder
        self.output_path = output_path
        self.house_data = None
        self.facility_data = None
        self.facility_types = None

    def load_facility_data(self):
        """讀取並合併所有類別的生活機能資料"""
        facility_files = {
            "food": "../food_data",
            "transport": "../transp_data",
            "education": "../edu_data",
            "entertainment": "../entert_data"
        }
        dataframes = []
        for category, folder in facility_files.items():
            files = [f for f in os.listdir(folder) if "位置" in f and f.endswith(".csv")]
            for file in files:
                df = pd.read_csv(os.path.join(folder, file), dtype=str)
                df["生活機能主項"] = category
                df.columns = ["名稱" if "名稱" in col else col for col in df.columns]
                df.columns = ["地址" if "地址" in col else col for col in df.columns]
                dataframes.append(df)
        self.facility_data = pd.concat(dataframes, ignore_index=True)
        self.facility_data.to_csv("生活機能資料.csv", encoding="utf-8-sig", index=False)
        self.facility_types = self.facility_data['機能項目'].unique()

    def load_house_data(self):
        """讀取房價資料"""
        self.house_data = pd.read_csv(self.house_data_path).dropna(subset=['緯度', '經度']).reset_index(drop=True)

    def process_facility_availability(self):
        """標記房屋是否鄰近特定生活機能"""
        facility_coords = self.facility_data[['緯度', '經度']].to_numpy()
        facility_tree = KDTree(facility_coords)
        search_radius = 10 / 111  # 1 度 ≈ 111 公里
        
        for facility in tqdm(self.facility_types, desc="標記房屋鄰近機能"):
            self.house_data[facility] = 0
        
        for i, house_row in tqdm(self.house_data.iterrows(), total=len(self.house_data), desc="處理房價資料"):
            nearby_facility_indices = facility_tree.query_ball_point([house_row['緯度'], house_row['經度']], search_radius)
            nearby_facilities = self.facility_data.iloc[nearby_facility_indices]['機能項目'].unique()
            for facility in nearby_facilities:
                self.house_data.at[i, facility] = 1

    def process_facility_distances(self):
        """計算房屋到最近生活機能的距離"""
        house_coords = self.house_data[['緯度', '經度']].to_numpy()
        max_distance_dict = {
            '速食店': 1, '商業登記餐廳': 1, '公司登記餐廳': 1, '超商': 1,  
            '台鐵': 5, '高捷': 2, '中捷': 2, '北捷': 2, '高鐵': 5, '公車': 2,  
            '大專院校': 5, '高中': 3, '國中': 3, '國小': 3,  
            '醫院': 3, '診所': 3, '衛生所': 3
        }
        
        for facility in tqdm(self.facility_types, desc="計算最近機能距離"):
            facility_subset = self.facility_data[self.facility_data['機能項目'] == facility][['緯度', '經度']].to_numpy()
            if len(facility_subset) == 0:
                continue
            facility_tree = KDTree(facility_subset)
            nearest_distances, _ = facility_tree.query(house_coords, k=1)
            nearest_distances_km = nearest_distances * 111
            max_valid_distance = max_distance_dict.get(facility, 5)
            nearest_distances_km[nearest_distances_km > max_valid_distance] = np.nan
            self.house_data[f'最近_{facility}_距離'] = nearest_distances_km

    def save_results(self):
        """儲存處理後的數據"""
        self.house_data.to_csv(self.output_path, encoding='utf-8-sig', index=False)
        print("✅ 房價與生活機能資料處理完成，已儲存至", self.output_path)

    def process_all(self):
        """執行所有步驟"""
        self.load_facility_data()
        self.load_house_data()
        self.process_facility_availability()
        self.process_facility_distances()
        self.save_results()

# ✅ 使用範例
if __name__ == "__main__":
    processor = FacilityProcessor(
        house_data_path='../2014年至2024年歷年實價登錄資料.csv',
        facility_folder='生活機能資料.csv',
        output_path='房價資料_含機能距離.csv'
    )
    processor.process_all()
