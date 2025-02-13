import os
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from scipy.spatial import KDTree
from lxml import etree
from processing.GeocodeFetcher import GeocodeFetcher

class GeoDataProcessor:
    def __init__(self, api_key, input_folder="raw/live_data", output_folder="processed_data"):
        self.geocoder = GeocodeFetcher(api_key)
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def process_education(self):
        """處理學校數據"""
        file_path = os.path.join(self.input_folder, "edu_data", "全台各級學校分布.csv")
        df = pd.read_csv(file_path, dtype=str)
        df = df.loc[:, ['學校級別', '學校名稱', '縣市別', '鄉鎮市區', '地址', '經度', '緯度']]
        df['地址'] = df['地址'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
        self.save_data(df, "education_processed.csv")

    def process_healthcare(self):
        """處理醫院與診所數據"""
        files = ["全台醫事機構_醫院.csv", "全台醫事機構_診所.csv"]
        for file in files:
            df = pd.read_csv(os.path.join(self.input_folder, "entert_data", file), dtype=str)
            df = df[['醫事機構名稱', '醫事機構種類', '地址']]
            df = self.process_geolocation(df, "地址")
            self.save_data(df, file.replace(".csv", "_processed.csv"))

    def process_food(self):
        """處理餐飲數據"""
        files = ["公司登記餐廳餐館.csv", "商業登記餐廳餐館.csv", "全國5大超商資料集.csv", "全國3大速食業資料集.csv"]
        for file in files:
            df = pd.read_csv(os.path.join(self.input_folder, "food_data", file), dtype=str)
            address_col = "公司地址" if "公司地址" in df.columns else "商業地址" if "商業地址" in df.columns else "分公司地址"
            df = df[["公司名稱", address_col]]
            df = self.process_geolocation(df, address_col)
            self.save_data(df, file.replace(".csv", "_processed.csv"))

    def process_transport(self):
        """處理交通站點數據"""
        transport_files = {
            "車站基本資料集.json": ["stationName", "stationAddrTw", "gps"],
            "taiwan-high-speed-rail.csv": ["name", "address", "lat", "lon"],
            "Northern_MRT.csv": ["station_name_tw", "address", "lat", "lon"],
            "Taichung_MRT.csv": ["車站中文", "地址", "緯度", "經度"],
            "Kaosiung_MRT.json": ["車站中文名稱", "街路門牌"]
        }
        for file, columns in transport_files.items():
            df = pd.read_json(os.path.join(self.input_folder, "transp_data", file)) if file.endswith(".json") else pd.read_csv(os.path.join(self.input_folder, "transp_data", file))
            df = df[columns]
            df = self.process_geolocation(df, columns[1])
            self.save_data(df, file.replace(".csv", "_processed.csv").replace(".json", "_processed.csv"))

        xml_file = os.path.join(self.input_folder, "transp_data", "公路客運站牌資料.xml")
        df_bus_stops = self.parse_bus_xml(xml_file)
        self.save_data(df_bus_stops, "公路客運站牌位置.csv")

    def parse_bus_xml(self, xml_file):
        """解析公車站點 XML"""
        tree = etree.parse(xml_file)
        root = tree.getroot()
        ns = {"ns": "https://ptx.transportdata.tw/standard/schema/"}
        def get_text(element, default="N/A"):
            return element.text if element is not None else default
        data = []
        for bus_stop in root.findall(".//ns:BusStop", namespaces=ns):
            data.append({
                "公路客運站名稱": get_text(bus_stop.find("ns:StopName/ns:Zh_tw", namespaces=ns)),
                "公路客運站地址": get_text(bus_stop.find("ns:StopAddress", namespaces=ns)),
                "經度": get_text(bus_stop.find("ns:StopPosition/ns:PositionLon", namespaces=ns)),
                "緯度": get_text(bus_stop.find("ns:StopPosition/ns:PositionLat", namespaces=ns))
            })
        return pd.DataFrame(data)

    def process_geolocation(self, df, address_col):
        """處理經緯度抓取，隨機間隔請求"""
        results = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="查詢經緯度"):
            lat, lng = self.geocoder.get_lat_lng(row[address_col])
            results.append((lat, lng))
            if (i + 1) % 100 == 0:
                time.sleep(random.randint(1, 5))
        df[["緯度", "經度"]] = pd.DataFrame(results, index=df.index)
        return df

    def save_data(self, df, filename):
        """儲存處理後的資料"""
        output_path = os.path.join(self.output_folder, filename)
        df.to_csv(output_path, encoding="utf-8-sig", index=False)
        print(f"✅ {filename} 處理完成，已存至 {output_path}")

    def process_all(self):
        """一次處理所有數據類型"""
        self.process_education()
        self.process_healthcare()
        self.process_food()
        self.process_transport()

# ✅ 使用範例
if __name__ == "__main__":
    API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
    processor = GeoDataProcessor(API_KEY)
    processor.process_all()
