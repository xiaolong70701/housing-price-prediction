import os
import glob
import pandas as pd
import requests
import time
import googlemaps
from tqdm import tqdm
import cn2an

class TaiwanHousingDataProcessor:
    def __init__(self, data_dir="raw/hist_price", output_dir="raw/processed_data", gmaps_api_key=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.gmaps = googlemaps.Client(key=gmaps_api_key) if gmaps_api_key else None
        
    def list_csv_files(self):
        return glob.glob(f"{self.data_dir}/*.csv")
    
    def list_info(self, csv_file):
        temp = pd.read_csv(csv_file)
        columns = list(temp.columns)
        num_of_cols = len(columns)
        return num_of_cols, columns
    
    def convert_csv_types(self, file_path: str, output_path: str = None) -> pd.DataFrame:
        dtype_conversion = {
            "主要建材": "category",
            "移轉層次": "category",
            "總樓層數": "str",
            "車位類別": "category",
            "電梯": "category",
        }

        try:
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到檔案: {file_path}")
        except Exception as e:
            raise RuntimeError(f"讀取 CSV 檔案時發生錯誤: {e}")
        
        for col, dtype in dtype_conversion.items():
            if col in df.columns:
                df[col] = df[col].replace(["", "nan", "None"], "無資料")
                if col == "建築完成年月":
                    df[col] = df[col].replace("無資料", pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(dtype)
            
        if output_path:
            df.to_csv(output_path, index=False)
        
        return df
    
    def batch_convert_csv(self):
        files = self.list_csv_files()
        for file_path in files:
            file_name = os.path.basename(file_path)
            output_path = os.path.join(self.output_dir, file_name)
            try:
                self.convert_csv_types(file_path, output_path)
                print(f"成功處理並存檔: {output_path}")
            except Exception as e:
                print(f"處理檔案 {file_path} 時發生錯誤: {e}")
    
    def get_coordinates(self, address):
        if not self.gmaps:
            raise ValueError("Google Maps API Key 未提供！")
        
        url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
        params = {
            "SingleLine": address,
            "f": "json",
            "outSR": '{"wkid":4326}',
            "outFields": "Addr_type,Match_addr,StAddr,City",
            "maxLocations": 1
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            location = data["candidates"][0]["location"]
            return location["y"], location["x"]
        else:
            return None, None
    
    def process_geolocation(self):
        files = self.list_csv_files()
        for file in tqdm(files, desc="處理地理座標資料"):
            df = pd.read_csv(file, dtype=str, low_memory=False)
            if "房屋地址" not in df.columns or "鄉鎮市區" not in df.columns:
                print(f"⚠️ {file} 缺少必要欄位，跳過...")
                continue
            
            df[["緯度", "經度"]] = df.progress_apply(
                lambda x: pd.Series(self.get_coordinates(str(x["房屋地址"]).strip())), axis=1
            )
            
            output_path = os.path.join(self.output_dir, os.path.basename(file))
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"✅ {file} 處理完成，已儲存至 {output_path}")
        print("🎉 所有地理座標資料處理完成！")
    
# 使用範例
if __name__ == "__main__":
    processor = TaiwanHousingDataProcessor(data_dir="hist_price", output_dir="processed_data", gmaps_api_key="YOUR_GOOGLE_MAPS_API_KEY")
    processor.batch_convert_csv()
    processor.process_geolocation()
