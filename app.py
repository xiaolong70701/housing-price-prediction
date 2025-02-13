import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import googlemaps
import json
from scipy.spatial import KDTree
from dotenv import load_dotenv

os.system("pip install xgboost")

# 載入 XGBoost 模型
with open("xgb_model_v4.pkl", "rb") as file:
    model = pickle.load(file)

# 確保模型的特徵名稱與輸入資料一致
expected_features = model.get_booster().feature_names

# 讀取 縣市別鄉鎮市區.json
with open("縣市別鄉鎮市區.json", "r", encoding="utf-8") as f:
    city_districts = json.load(f)

# 載入標籤編碼對照表
label_map = pd.read_csv("機器學習-標籤編碼對照表v2.csv")

# 解析標籤編碼對照表，建立映射字典
encoding_dict = {}
for column in label_map.columns:
    if "_原內容" in column:
        base_name = column.replace("_原內容", "")
        encoding_dict[base_name] = dict(zip(label_map[column], label_map[f"{base_name}_對應編碼"]))

# 載入 Google Maps API Key
load_dotenv()
# ✅ 使用環境變數獲取 API_KEY
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# ✅ 確保 API_KEY 存在
if not API_KEY:
    st.error("❌ 缺少 Google Maps API Key，請在 .env 檔案中設定！")
    st.stop()

gmaps = googlemaps.Client(key=API_KEY)

# 讀取生活機能資料
facility_data = pd.read_csv('生活機能資料.csv')
facility_data = facility_data.dropna(subset=['緯度', '經度'])

# 讀取平均單價 JSON
with open('average_price_per_sqm.json', 'r', encoding='utf-8') as f:
    avg_price_data = json.load(f)

# 設定不同機能的最大有效範圍（公里）
MAX_DISTANCE_DICT = {
    '速食店': 1, '商業登記餐廳': 1, '公司登記餐廳': 1, '超商': 1,  
    '台鐵': 5, '高捷': 2, '中捷': 2, '北捷': 2, '高鐵': 10, '公車': 2,  
    '大專院校': 5, '高中': 3, '國中': 3, '國小': 3,  
    '醫院': 3, '診所': 3, '衛生所': 3
}

# 計算最近機能距離
def get_nearest_distances(address):
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return {f'最近_{key}_距離': np.nan for key in MAX_DISTANCE_DICT.keys()}

    lat_lng = geocode_result[0]['geometry']['location']
    lat, lng = lat_lng['lat'], lat_lng['lng']

    result = {}
    for facility, max_dist in MAX_DISTANCE_DICT.items():
        subset = facility_data[facility_data['機能項目'] == facility][['緯度', '經度']].to_numpy()

        if len(subset) > 0:
            facility_tree = KDTree(subset)
            nearest_distance, _ = facility_tree.query([lat, lng], k=1)
            nearest_distance_km = nearest_distance * 111  
            
            if nearest_distance_km > max_dist:
                nearest_distance_km = np.nan

            result[f'最近_{facility}_距離'] = nearest_distance_km
        else:
            result[f'最近_{facility}_距離'] = np.nan

    return result

# 取得 log_鄰近區域平均單價
def get_avg_price(transaction_year, district):
    key = f"{transaction_year}_{district}"
    log_price = np.log(avg_price_data.get(key, np.nan))
    return log_price

# 進行標籤編碼轉換
def encode_categorical_features(df):
    for col in encoding_dict:
        if col in df.columns:
            df[col] = df[col].map(encoding_dict[col]).astype("Int64")
    return df

# Streamlit UI
st.title("房價預測系統")

st.sidebar.header("請輸入房屋資訊")

# **新增地址輸入框**
address_input = st.sidebar.text_input("地址（可選填，如果知道房屋地址）")

# 選擇縣市
city = st.sidebar.selectbox("縣市別", list(city_districts.keys()))

# 根據選擇的縣市，自動更新鄉鎮市區選單
district = st.sidebar.selectbox("鄉鎮市區", city_districts[city])

# 其他輸入項目
transaction_year = st.sidebar.number_input("交易年份", min_value=2000, max_value=2025, value=2023)
house_age = st.sidebar.number_input("房屋年齡", min_value=0, max_value=100, value=20)
area_tsubo = st.sidebar.number_input("交易面積（坪）", min_value=5.0, max_value=500.0, value=30.0)
area_sqm = area_tsubo * 3.30578  # 坪轉換為平方公尺
parking_type = st.sidebar.selectbox("車位類別", encoding_dict["車位類別"].keys())
management = st.sidebar.selectbox("有無管理組織", encoding_dict["有無管理組織"].keys())
elevator = st.sidebar.selectbox("有無電梯", encoding_dict["電梯"].keys())
building_type = st.sidebar.selectbox("建物型態", encoding_dict["建物型態"].keys())
rooms = st.sidebar.number_input("建物現況格局-房", min_value=1, max_value=10, value=3)
living_rooms = st.sidebar.number_input("建物現況格局-廳", min_value=0, max_value=5, value=1)
bathrooms = st.sidebar.number_input("建物現況格局-衛", min_value=1, max_value=5, value=1)
floor_ratio = st.sidebar.number_input("樓高比", min_value=0.1, max_value=1.5, value=0.8)

if st.sidebar.button("預測價格"):
    # **如果使用者有輸入地址，就用該地址，否則用縣市+鄉鎮市區**
    address = address_input if address_input.strip() else f"{city}{district}"

    nearest_distances = get_nearest_distances(address)
    log_avg_price = get_avg_price(transaction_year, district)
    high_price_index = 1 if log_avg_price > 12 else 0

    input_data = {
        '縣市別': city, '鄉鎮市區': district, '交易年份': transaction_year,
        '房屋年齡': house_age, '交易面積': area_sqm, '車位類別': parking_type,
        '有無管理組織': management, '電梯': elevator, '建物型態': building_type,
        '建物現況格局-房': rooms, '建物現況格局-廳': living_rooms, '建物現況格局-衛': bathrooms,
        '樓高比': floor_ratio, '高價指標': high_price_index,
        'log_鄰近區域平均單價': log_avg_price
    }
    input_data.update(nearest_distances)

    df = pd.DataFrame([input_data])
    df = encode_categorical_features(df)
    df = df[expected_features]  # 確保特徵順序正確

    log_predicted_price = model.predict(df)[0]
    predicted_price_per_sqm = np.exp(log_predicted_price)  # 還原 log 轉換
    predicted_total_price = predicted_price_per_sqm * area_sqm  # 計算總價

    st.subheader(f"預測單價： {predicted_price_per_sqm:.3f} 元/平方公尺")
    st.subheader(f"預測總價： {predicted_total_price:,.0f} 元")
