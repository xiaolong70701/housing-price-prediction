import googlemaps
import pandas as pd

class GeocodeFetcher:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key)
    
    def get_lat_lng(self, address, district=None):
        try:
            # 第一次嘗試用原始地址查詢
            geocode_result = self.gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]["geometry"]["location"]
                return round(location["lat"], 4), round(location["lng"], 4)

            # 如果查不到，再用「鄉鎮市區 + 房屋地址」重新查詢
            if pd.notna(district) and pd.notna(address):
                full_address = f"{district} {address}".strip()
                geocode_result = self.gmaps.geocode(full_address)
                if geocode_result:
                    location = geocode_result[0]["geometry"]["location"]
                    return round(location["lat"], 4), round(location["lng"], 4)

        except Exception as e:
            print(f"❌ 地址轉換失敗：{address}, Error: {e}")
        return None, None