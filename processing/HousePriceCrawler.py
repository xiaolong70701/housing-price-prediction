import os
import time
import zipfile
import requests
import pandas as pd
import cn2an
from tqdm import tqdm

class TaiwanHousingPrice:
    def __init__(self, zip_folder='raw/zip_files', price_folder='raw/price_files', hist_folder='raw/hist_price'):
        self.zip_folder = zip_folder
        self.price_folder = price_folder
        self.hist_folder = hist_folder
        self.locToLetter = self._define_location_codes()
        self.letterToLoc = {v: k for k, v in self.locToLetter.items()}
        
        # 建立資料夾
        for folder in [self.zip_folder, self.price_folder, self.hist_folder]:
            os.makedirs(folder, exist_ok=True)
    
    def _define_location_codes(self):
        location_str = """台北市 A 苗栗縣 K 花蓮縣 U
        台中市 B 台中縣 L 台東縣 V
        基隆市 C 南投縣 M 澎湖縣 X
        台南市 D 彰化縣 N 陽明山 Y
        高雄市 E 雲林縣 P 金門縣 W
        新北市 F 嘉義縣 Q 連江縣 Z
        宜蘭縣 G 台南縣 R 嘉義市 I
        桃園市 H 高雄縣 S 新竹市 O
        新竹縣 J 屏東縣 T"""
        return dict(zip(location_str.split()[::2], location_str.lower().split()[1::2]))

    def download_zip(self, year, season):
        if year > 1000:
            year -= 1911  # 轉換為民國年

        url = f"https://plvr.land.moi.gov.tw//DownloadSeason?season={year}S{season}&type=zip&fileName=lvr_landcsv.zip"
        res = requests.get(url)
        fname = f'{self.zip_folder}/{year}{season}.zip'
        
        with open(fname, 'wb') as f:
            f.write(res.content)
        
        return fname
    
    def extract_zip(self, zip_file, year, season):
        folder = f'{self.price_folder}/{year}{season}_price'
        os.makedirs(folder, exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(folder)
        
        time.sleep(5)  # 避免頻繁讀取錯誤
        return folder

    def fetch_data(self, start_year=103, end_year=114):
        total_tasks = (end_year - start_year) * 4
        with tqdm(total=total_tasks, desc="Downloading and Extracting Data") as pbar:
            for year in range(start_year, end_year):
                for season in range(1, 5):
                    zip_file = self.download_zip(year, season)
                    self.extract_zip(zip_file, year, season)
                    pbar.update(1)
    
    def parse_date(self, date):
        date = str(date)
        if date.isdigit() and len(date) == 7:
            return str(int(date[:3]) + 1911), date[3:5], date[5:]
        return '', '', ''
    
    def safe_cn2an(self, value):
        try:
            return cn2an.cn2an(value, 'smart')
        except ValueError:
            return None

    def process_data(self, location):
        if location not in self.locToLetter:
            raise ValueError("Invalid location name")
        
        dirs = [d for d in os.listdir(self.price_folder) if d.endswith('price')]
        dfs = []
        
        for d in dirs:
            file_path = os.path.join(self.price_folder, d, self.locToLetter[location] + '_lvr_land_a.csv')
            df = pd.read_csv(file_path, index_col=False, low_memory=False)
            df['季度'] = d[3]
            df['交易年月日'] = df['交易年月日'].astype(str)
            df.loc[df['交易年月日'].str.len() != 7, '交易年月日'] = d[:3] + '0000'
            dfs.append(df.iloc[1:])
        
        df = pd.concat(dfs, sort=True)
        df[['交易年份', '交易月份', '交易日期']] = df['交易年月日'].apply(self.parse_date).apply(pd.Series)
        df[['建築完成年', '建築完成月', '建築完成日']] = df['建築完成年月'].apply(self.parse_date).apply(pd.Series)
        
        for col in ['交易年份', '交易月份', '交易日期', '建築完成年', '建築完成月', '建築完成日']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df['房屋年齡'] = df['交易年份'] - df['建築完成年']
        df = df[df['房屋年齡'] <= 150]
        df['建物型態'] = df['建物型態'].str.split('(').str[0]
        df['總樓層數'] = df['總樓層數'].str.replace('層', '', regex=False).apply(self.safe_cn2an)
        df.dropna(subset=['總樓層數'], inplace=True)
        df['總樓層數'] = df['總樓層數'].astype(int)
        
        return df
    
    def save_to_csv(self, location):
        df = self.process_data(location)
        file_path = f'{self.hist_folder}/{location}_歷年實價登錄資料.csv'
        df.to_csv(file_path, encoding='utf-8-sig', index=False)

# 使用範例
if __name__ == "__main__":
    zip_folder="data/zip_files"
    price_folder="data/price_files"
    hist_folder="data/hist_price"
    
    crawler = TaiwanHousingPrice(zip_folder=zip_folder, price_folder=price_folder, hist_folder=hist_folder)
    crawler.fetch_data(103, 114)
