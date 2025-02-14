import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import googlemaps
import json
from scipy.spatial import KDTree
from dotenv import load_dotenv
import geopandas as gpd
import lightgbm as lgb
import pydeck as pdk
from draw import draw_map, draw_bar

# 讀取 .env（本地環境）
load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

st.set_page_config(page_title="房價預測系統", layout="wide")

if not API_KEY and "GOOGLE_MAPS_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]

if not API_KEY:
    st.error("❌ 缺少 Google Maps API Key，請在 .env 或 Streamlit Secrets 設定！")
    st.stop()

gmaps = googlemaps.Client(key=API_KEY)

with open('xgb_model_v4.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
# 讀取 LightGBM 模型
with open('lgb_model_low.pkl', 'rb') as f:
    model_low = pickle.load(f)
with open('lgb_model_mid.pkl', 'rb') as f:
    model_mid = pickle.load(f)
with open('lgb_model_high.pkl', 'rb') as f:
    model_high = pickle.load(f)

# 確保特徵順序一致
feature_order = model_mid.feature_name_

# 讀取標籤編碼對照表
label_map = pd.read_csv("機器學習-標籤編碼對照表v2.csv")
encoding_dict = {}
for column in label_map.columns:
    if "_原內容" in column:
        base_name = column.replace("_原內容", "")
        mapping = dict(zip(label_map[column], label_map[f"{base_name}_對應編碼"]))
        encoding_dict[base_name] = {k: v for k, v in mapping.items() if pd.notna(k)}

# 讀取地理資訊（鄉鎮邊界）
import geopandas as gpd
import pandas as pd

@st.cache_data
def load_gdf():
    # 讀取 SHP 檔案
    gdf_raw = gpd.read_file('mapData_20250129/TOWN_MOI_1131028.shp', encoding='utf-8')
    gdf_raw["COUNTYNAME"] = gdf_raw["COUNTYNAME"].str.strip()
    gdf_raw["TOWNNAME"] = gdf_raw["TOWNNAME"].str.strip()

    # 建立 `place` 欄位
    gdf_raw["place"] = gdf_raw["TOWNNAME"]

    try:
        place_df = pd.read_csv('Place.csv')
        if "place" in place_df.columns:
            gdf = pd.merge(gdf_raw, place_df, on="place", how="left")
        else:
            gdf = gdf_raw.copy()
    except FileNotFoundError:
        gdf = gdf_raw.copy()

    if "COUNTYNAME" not in gdf.columns:
        gdf["COUNTYNAME"] = gdf_raw["COUNTYNAME"]  # 如果 `merge()` 移除了，補回來

    gdf = gdf[['COUNTYNAME', 'place', 'geometry']]
    return gdf


gdf = load_gdf()

# 讀取生活機能資料
facility_data = pd.read_csv('生活機能資料.csv').dropna(subset=['緯度', '經度'])

# 讀取縣市 & 鄉鎮市區對應表
with open("縣市別鄉鎮市區.json", "r", encoding="utf-8") as f:
    city_districts = json.load(f)

# 讀取平均單價 JSON
with open('average_price_per_sqm.json', 'r', encoding='utf-8') as f:
    avg_price_data = json.load(f)

# 設定機能最大有效距離
MAX_DISTANCE_DICT = {
    '速食店': 1, '商業登記餐廳': 1, '公司登記餐廳': 1, '超商': 1,
    '台鐵': 5, '高捷': 2, '中捷': 2, '北捷': 2, '高鐵': 10, '公車': 2,
    '大專院校': 5, '高中': 3, '國中': 3, '國小': 3, '醫院': 3, '診所': 3, '衛生所': 3
}

# 計算最近機能距離
def get_nearest_distances(address):
    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return {f'最近_{key}_距離': np.nan for key in MAX_DISTANCE_DICT.keys()}

    lat, lng = geocode_result[0]['geometry']['location'].values()
    result = {}

    for facility, max_dist in MAX_DISTANCE_DICT.items():
        subset = facility_data[facility_data['機能項目'] == facility][['緯度', '經度']].to_numpy()
        if len(subset) > 0:
            facility_tree = KDTree(subset)
            nearest_distance, _ = facility_tree.query([lat, lng], k=1)
            nearest_distance_km = nearest_distance * 111
            result[f'最近_{facility}_距離'] = nearest_distance_km if nearest_distance_km <= max_dist else np.nan
        else:
            result[f'最近_{facility}_距離'] = np.nan

    return result

# 取得 log_鄰近區域平均單價
def get_avg_price(transaction_year, district):
    transaction_year = str(transaction_year)
    
    # 嘗試查詢當前年份的價格
    log_price = avg_price_data.get(transaction_year, {}).get(district, None)
    
    # 如果 log_price 為空，則繼續查詢上一年
    while log_price is None and int(transaction_year) > 2000:  # 假設資料的年份不會低於 2000
        transaction_year = str(int(transaction_year) - 1)
        log_price = avg_price_data.get(transaction_year, {}).get(district, None)
    
    # 若依然找不到資料，則返回 None 或預設值
    if log_price is None:
        print(f"無法找到 {district} 在 {transaction_year} 或之前的房價資料")
    
    return log_price


# 標籤編碼
def encode_categorical_features(df):
    for col in encoding_dict:
        if col in df.columns:
            df[col] = df[col].map(encoding_dict[col]).astype("Int64")
    return df

# Streamlit UI
st.markdown("""
    <style>
    .main-content {
        max-width: 900px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# 使用者選擇模型
with st.container():
    st.title("🏠 房價預測模型")
    model_choice = st.radio("選擇預測模型", ["XGBoost", "LightGBM"])

    address_input = st.text_input("地址（可選填，如果知道房屋地址）")

    # 優化輸入欄位顯示，分為兩個區塊
    col1, col2 = st.columns(2)  # 將兩欄設為 50% 寬度

    # 在第一欄輸入欄位
    with col1:
        city = st.selectbox("縣市別", list(city_districts.keys()))
        transaction_year = st.number_input("交易年份", min_value=2000, max_value=2025, value=2023)
        house_age = st.number_input("房屋年齡", min_value=0.1, max_value=100.0, value=20.0)
        parking_type = st.selectbox("車位類別", encoding_dict["車位類別"].keys())
        management = st.selectbox("有無管理組織", encoding_dict["有無管理組織"].keys())
        elevator = st.selectbox("有無電梯", encoding_dict["電梯"].keys())
        rooms = st.number_input("建物現況格局-房", min_value=1, max_value=10, value=3)

    # 在第二欄輸入欄位
    with col2:
        district = st.selectbox("鄉鎮市區", city_districts[city])
        area_tsubo = st.number_input("交易面積（坪）", min_value=5.0, max_value=500.0, value=30.0)
        area_sqm = area_tsubo * 3.30578  # 將坪數轉換為平方公尺
        building_type = st.selectbox("建物型態", encoding_dict["建物型態"].keys())
        living_rooms = st.number_input("建物現況格局-廳", min_value=0, max_value=5, value=1)
        bathrooms = st.number_input("建物現況格局-衛", min_value=1, max_value=5, value=1)
        at_floor = st.number_input("所在樓層", min_value=1, max_value=100, value=1)
        total_floor = st.number_input("總樓層", min_value=1, max_value=100, value=1)

    # 計算樓層比例
    floor_ratio = at_floor / total_floor


    # transaction_year = st.sidebar.number_input("交易年份", min_value=2000, max_value=2025, value=2023)
    # house_age = st.sidebar.number_input("房屋年齡", min_value=0.1, max_value=100.0, value=20.0)
    # area_tsubo = st.sidebar.number_input("交易面積（坪）", min_value=5.0, max_value=500.0, value=30.0)
    # area_sqm = area_tsubo * 3.30578
    # parking_type = st.sidebar.selectbox("車位類別", encoding_dict["車位類別"].keys())
    # management = st.sidebar.selectbox("有無管理組織", encoding_dict["有無管理組織"].keys())
    # elevator = st.sidebar.selectbox("有無電梯", encoding_dict["電梯"].keys())
    # building_type = st.sidebar.selectbox("建物型態", encoding_dict["建物型態"].keys())
    # rooms = st.sidebar.number_input("建物現況格局-房", min_value=1, max_value=10, value=3)
    # living_rooms = st.sidebar.number_input("建物現況格局-廳", min_value=0, max_value=5, value=1)
    # bathrooms = st.sidebar.number_input("建物現況格局-衛", min_value=1, max_value=5, value=1)
    # at_floor = st.sidebar.number_input("所在樓層", min_value=1, max_value=100, value=1)
    # total_floor = st.sidebar.number_input("總樓層", min_value=1, max_value=100, value=1)
    # floor_ratio = at_floor / total_floor

    if st.button("預測價格"):
        address = address_input if address_input.strip() else f"{city}{district}"

        predicted_prices = {}  # 存放每個區的預測結果

        for district_name in city_districts[city]:
            nearest_distances = get_nearest_distances(f"{city}{district_name}")
            log_avg_price = get_avg_price(transaction_year, district)
            high_price_index = 1 if log_avg_price > 12 else 0

            input_data = {
                '縣市別': city, '鄉鎮市區': district_name, '交易年份': transaction_year,
                '房屋年齡': house_age, '交易面積': area_sqm, '車位類別': parking_type,
                '有無管理組織': management, '電梯': elevator, '建物型態': building_type,
                '建物現況格局-房': rooms, '建物現況格局-廳': living_rooms, '建物現況格局-衛': bathrooms,
                '樓高比': floor_ratio, '高價指標': high_price_index,
                'log_鄰近區域平均單價': log_avg_price
            }
            input_data.update(nearest_distances)

            df = pd.DataFrame([input_data])
            df = encode_categorical_features(df)
            df = df[feature_order]

            # 使用選擇的模型進行預測
            if model_choice == "LightGBM":
                low_pred_log = model_low.predict(df)[0]
                mid_pred_log = model_mid.predict(df)[0]
                high_pred_log = model_high.predict(df)[0]

                # 確保 95% 分位數不低於 50% 分位數
                if high_pred_log < mid_pred_log:
                    high_pred_log = mid_pred_log

                low_pred = np.exp(low_pred_log)
                mid_pred = np.exp(mid_pred_log)
                high_pred = np.exp(high_pred_log)

                predicted_total_price_low = low_pred * area_sqm
                predicted_total_price_mid = mid_pred * area_sqm
                predicted_total_price_high = high_pred * area_sqm

                predicted_prices[district_name] = {
                    "price_wan": round(predicted_total_price_mid / 10_000, 3),
                    "unit_price_wan": round(mid_pred / 10_000, 3)
                }
            else:
                # 如果選擇的是 XGBoost 模型
                log_pred = xgb_model.predict(df)[0]
                pred_price_per_sqm = np.exp(log_pred)
                # print(f"每平方公尺單價: {pred_price_per_sqm}，總共 {area_sqm} 平方公尺。")
                predicted_total_price = pred_price_per_sqm * area_sqm

                predicted_prices[district_name] = {
                    "price_wan": predicted_total_price,
                    "unit_price_wan": round(pred_price_per_sqm / 10_000, 3)
                }

        house_gdf = gdf[gdf["COUNTYNAME"] == city].copy()
        house_gdf["price_wan"] = house_gdf["place"].map(lambda x: predicted_prices.get(x, {}).get("price_wan", None))
        house_gdf["unit_price_wan"] = house_gdf["place"].map(lambda x: predicted_prices.get(x, {}).get("unit_price_wan", None))
        
        target_total_price = predicted_prices[district]['price_wan']
        target_unit_price = target_total_price / area_tsubo

        st.subheader("📊 預測結果")
        if model_choice == "XGBoost":
            st.write(f"💰 **預測單價**： {target_unit_price:,.0f} 元/坪")
            st.write(f"💰 **預測總價**： {target_total_price:,.0f} 元")
        else:
            st.write(f"💰 **預測區間**： {predicted_total_price_low:,.0f} 元 ~ {predicted_total_price_high:,.0f} 元")
            st.write(f"💰 **中位數預測**： {predicted_total_price_mid:,.0f} 元")


        st.subheader("📍 預測地圖")
        map_visual = draw_map(house_gdf, city, district)
        st.pydeck_chart(map_visual)

        # **新增色條來顯示房價範圍**
        st.subheader("🎨 房價對應色條")
        min_price = min([v["price_wan"] for v in predicted_prices.values() if v["price_wan"] is not None])
        max_price = max([v["price_wan"] for v in predicted_prices.values() if v["price_wan"] is not None])

        fig = draw_bar(predicted_prices[district]["price_wan"], min_price, max_price, city)
        st.pyplot(fig)
