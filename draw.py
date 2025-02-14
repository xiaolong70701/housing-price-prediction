import pydeck as pdk
import pandas as pd
import json
import numpy as np
import leafmap.colormaps as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from leafmap.common import hex_to_rgb
from matplotlib.font_manager import FontProperties as font

# 設定字型（確保 Matplotlib 支援中文）
font1 = font(fname="fonts/NotoSansTC-Regular.ttf")

def load_city_districts():
    """ 讀取 `縣市別鄉鎮市區.json` 並回傳字典 """
    with open("縣市別鄉鎮市區.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_districts(city, target_district=None):
    """ 查詢 `city` 的所有鄉鎮市區，可排除 `target_district` """
    city_districts_dict = load_city_districts()
    if city in city_districts_dict:
        districts = city_districts_dict[city]
        if target_district:
            districts = [d for d in districts if d != target_district]
        return districts
    return []

def adjust_color(color, factor=0.5):
    """ 調整顏色深度，使其更暗 """
    return [max(int(c * factor), 0) for c in color]  # 限制最低為 0，確保不溢出

def draw_map(gdf, city, target_district):
    """ 根據 `city` 的房價數據繪製地圖，使用 `leafmap.colormaps` 來產生漸變色 """

    # ✅ 只保留該 `city` 的資料
    city_gdf = gdf[gdf["COUNTYNAME"] == city].copy()

    # ✅ 讀取城市經緯度資訊
    city_df = pd.read_csv('City_map.csv')
    try:
        city_info = city_df[city_df['city'] == city].reset_index()
        latitude = city_info['latitude'][0]
        longitude = city_info['longitude'][0]
        zoom = city_info['zoom'][0]
    except:
        latitude, longitude, zoom = 23.5, 121, 7
        print(f"⚠️ 找不到 `{city}`，使用預設中心點")

    # ✅ 計算房價範圍，確保漸變顏色映射正確
    min_price = city_gdf["price_wan"].fillna(0).min()
    max_price = city_gdf["price_wan"].fillna(0).max()

    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        min_price, max_price = 0, 1  # 避免錯誤

    # ✅ 決定 `n_colors`（該 `city` 內的鄉鎮市區數量）
    districts = get_districts(city)
    n_colors = len(districts) + 1  # 多加一個確保 `target_district` 有顏色

    # ✅ 取得顏色映射
    palettes = cm.list_colormaps()
    palette = palettes[2]  # 這裡選擇第 2 個 colormap，可自行修改
    colors = cm.get_palette(palette, n_colors)
    colors = [hex_to_rgb(c) for c in colors]  # 轉換 HEX 為 RGB
    colors = [adjust_color(c, factor=1) for c in colors]  # 調淡所有顏色

    # ✅ 設定 `target_district` 為最深藍色，其他鄉鎮依房價排序
    target_color = adjust_color(colors[-1], factor=0.5)  # 讓 target_district 更深

    # ✅ 根據 `price_wan` 排序，確保最低房價使用最淺的顏色
    city_gdf = city_gdf.sort_values(by="price_wan", ascending=True).reset_index()

    def assign_color(row):
        if row["place"] == target_district:
            return target_color  # `target_district` 使用最深藍色
        if pd.notna(row["price_wan"]):
            index = int(((row["price_wan"] - min_price) / (max_price - min_price)) * (n_colors - 1))
            index = min(index, n_colors - 1)  # 確保 index 不超出範圍
            return colors[index]  # 依房價映射顏色
            
        return [200, 200, 200]  # 沒有價格的區域顯示灰色

    city_gdf[["R", "G", "B"]] = city_gdf.apply(lambda row: pd.Series(assign_color(row)), axis=1)
    
    # ✅ 建立 GeoJsonLayer
    geojson = pdk.Layer(
        "GeoJsonLayer",
        city_gdf,
        pickable=True,
        opacity=0.5,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="[R, G, B]",
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": "【預測房價】<br><b>地區:</b> {place}<br><b>總房價:</b> {price_wan}萬<br>",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    return pdk.Deck(
        layers=[geojson],
        initial_view_state=pdk.ViewState(
            latitude=latitude, 
            longitude=longitude, 
            zoom=zoom, 
            max_zoom=20
        ),
        map_style="light",
        tooltip=tooltip,
    )

def draw_bar(price, min_price, max_price, city):
    """ 繪製房價對應的顏色標示 """
    palettes = cm.list_colormaps()
    palette = palettes[2]  # 選擇與地圖一致的 colormap
    
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    
    cmap = plt.get_cmap(palette)
    norm = mpl.colors.Normalize(vmin=min_price, vmax=max_price)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, 
        orientation='horizontal'
    )

    ax.set_title(f"{city} 房價預測區間", fontproperties=font1, fontsize=12)

    mid = (min_price + max_price) / 2
    price_rate = mid + (price - mid) * 0.995
    cbar.ax.plot([price_rate, price_rate], [0, 1], 'red', linewidth=2)
    cbar.ax.plot([price_rate, price_rate], [0.9, 1], color='red', marker='v', linewidth=0.10)
    cbar.ax.plot([price_rate, price_rate], [0, 0.1], color='red', marker='^', linewidth=0.10)

    return fig
