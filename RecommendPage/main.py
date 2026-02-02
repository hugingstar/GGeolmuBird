import os
import pandas as pd
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import date

app = FastAPI(title="GGeolmuBird")
templates = Jinja2Templates(directory="templates")

# 1. 데이터 루트 경로 (WSL 경로)
DATA_ROOT_DIR = r"\\wsl.localhost\Ubuntu\home\yslee\TRADE\Data\DTAS\Data"

# 2. 종목 코드 매핑 파일 루트 경로 (Windows 경로)
STOCK_LIST_ROOT = r"C:\Users\User\PycharmProjects\GGeolmuBird\RecommendPage\market"

MARKETS = ["KOSPI", "KOSDAQ", "NASDAQ", "NYSE"]

ALL_COLUMNS = [ "Name", "Open", "High", "Low", "Close", "Volume", "Close_diff_first", "Close_diff_second",
    "Close_rate_first", "Close_rate_second", "CurrencyVolume", "obv_change", "OBV",
    "MA5", "MA50", "MA60", "MA120", "MA200", "MA224", "MA20", "STD20", "BB_Upper", "BB_Lower",
    "RSI", "RSI_rate_first", "RSI_rate_second", "RSI2", "RSI3", "RSI4", "RSI5", "RSI6", "RSI7", "RSI8", "RSI9",
    "RSI_Signal", "RSI_BullDiv", "RSI_BearDiv", "RSI_Hidden_BullDiv", "RSI_Hidden_BearDiv",
    "RSI_Signal2", "RSI_BullDiv2", "RSI_BearDiv2", "RSI_Hidden_BullDiv2", "RSI_Hidden_BearDiv2",
    "RSI_Signal3", "RSI_BullDiv3", "RSI_BearDiv3", "RSI_Hidden_BullDiv3", "RSI_Hidden_BearDiv3",
    "RSI_Signal4", "RSI_BullDiv4", "RSI_BearDiv4", "RSI_Hidden_BullDiv4", "RSI_Hidden_BearDiv4",
    "RSI_Signal5", "RSI_BullDiv5", "RSI_BearDiv5", "RSI_Hidden_BullDiv5", "RSI_Hidden_BearDiv5",
    "RSI_Signal6", "RSI_BullDiv6", "RSI_BearDiv6", "RSI_Hidden_BullDiv6", "RSI_Hidden_BearDiv6",
    "RSI_Signal7", "RSI_BullDiv7", "RSI_BearDiv7", "RSI_Hidden_BullDiv7", "RSI_Hidden_BearDiv7",
    "RSI_Signal8", "RSI_BullDiv8", "RSI_BearDiv8", "RSI_Hidden_BullDiv8", "RSI_Hidden_BearDiv8",
    "RSI_Signal9", "RSI_BullDiv9", "RSI_BearDiv9", "RSI_Hidden_BullDiv9", "RSI_Hidden_BearDiv9",
    "CCI", "CCI_rate_first", "CCI_rate_second", "CCI2", "CCI3", "CCI4", "CCI5", "CCI6", "CCI7", "CCI8", "CCI9",
    "CCI_Signal", "CCI_BullDiv", "CCI_BearDiv", "CCI_Hidden_BullDiv", "CCI_Hidden_BearDiv",
    "CCI_Signal2", "CCI_BullDiv2", "CCI_BearDiv2", "CCI_Hidden_BullDiv2", "CCI_Hidden_BearDiv2",
    "CCI_Signal3", "CCI_BullDiv3", "CCI_BearDiv3", "CCI_Hidden_BullDiv3", "CCI_Hidden_BearDiv3",
    "CCI_Signal4", "CCI_BullDiv4", "CCI_BearDiv4", "CCI_Hidden_BullDiv4", "CCI_Hidden_BearDiv4",
    "CCI_Signal5", "CCI_BullDiv5", "CCI_BearDiv5", "CCI_Hidden_BullDiv5", "CCI_Hidden_BearDiv5",
    "CCI_Signal6", "CCI_BullDiv6", "CCI_BearDiv6", "CCI_Hidden_BullDiv6", "CCI_Hidden_BearDiv6",
    "CCI_Signal7", "CCI_BullDiv7", "CCI_BearDiv7", "CCI_Hidden_BullDiv7", "CCI_Hidden_BearDiv7",
    "CCI_Signal8", "CCI_BullDiv8", "CCI_BearDiv8", "CCI_Hidden_BullDiv8", "CCI_Hidden_BearDiv8",
    "CCI_Signal9", "CCI_BullDiv9", "CCI_BearDiv9", "CCI_Hidden_BullDiv9", "CCI_Hidden_BearDiv9",
    "MACD", "MACD_Base", "MACD_Hist", "MACD_Positive", "MACD_Signal",
    "PDI", "MDI", "ADX", "DX", "ADXR", "DMI_Trend", "DMI_BullCross", "DMI_BearCross",
    "High_watermark", "Drawdown",
    "Sell_Signal", "Sell_Signal2", "Sell_Signal3", "Sell_Signal4", "Sell_Signal5",
    "Sell_Signal6", "Sell_Signal7", "Sell_Signal8", "Sell_Signal9"
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "all_columns": ALL_COLUMNS,
        "markets": MARKETS,
        "tables": {},
        "counts": {"bull": 0, "bear": 0, "sell": 0},
        "chart_data": "{}", # 빈 데이터
        "selected_date": date.today().isoformat(),
        "selected_market": "KOSPI",
        "selected_columns": ["Name", "Open", "Close", "Volume"] 
    })

@app.post("/", response_class=HTMLResponse)
async def search_data(
    request: Request,
    target_date: str = Form(...),
    market: str = Form(...),
    selected_cols: list = Form(None)
):
    if not selected_cols:
        selected_cols = ALL_COLUMNS

    stock_map = {}
    map_file_path = os.path.join(STOCK_LIST_ROOT, market, "stock_list.csv")
    
    if os.path.exists(map_file_path):
        try:
            ref_df = pd.read_csv(map_file_path, dtype={'Code': str, 'Symbol': str})
            code_col = 'Code' if 'Code' in ref_df.columns else 'Symbol'
            stock_map = dict(zip(ref_df['Name'], ref_df[code_col]))
        except Exception as e:
            print(f"Error loading stock list: {e}")

    folder_path = os.path.join(DATA_ROOT_DIR, market, "B1Sheet", target_date)
    
    data_types = {
        "bull": "df_bull",
        "bear": "df_bear",
        "sell": "df_sell"
    }
    
    tables_html = {} 
    counts = {}
    chart_data = {} # 차트 데이터 저장소

    for key, filename_base in data_types.items():
        file_path = os.path.join(folder_path, f"{filename_base}.csv")
        
        # 차트 데이터 초기화
        chart_data[key] = []

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, encoding='cp949')
                
                # 1. 차트용 데이터 추출 (Name, RSI5, CCI5)
                # 컬럼이 존재하는지 확인 후 데이터 추출 (NaN은 0으로 처리)
                if set(['Name', 'RSI5', 'CCI5']).issubset(df.columns):
                    chart_subset = df[['Name', 'RSI5', 'CCI5']].fillna(0)
                    chart_data[key] = chart_subset.to_dict(orient='records')

                # 2. 테이블 데이터 처리
                # 데이터 필터링
                valid_cols = [c for c in selected_cols if c in df.columns]
                df_filtered = df[valid_cols].copy()

                # 종목 수 카운팅
                counts[key] = len(df_filtered)
                
                # 링크 생성
                if 'Name' in df_filtered.columns:
                    def create_link(name):
                        code = stock_map.get(name)
                        if not code: return name
                        if market in ["KOSPI", "KOSDAQ"]:
                            return f'<a href="https://finance.naver.com/item/main.naver?code={code}" target="_blank" class="stock-link">{name}</a>'
                        else:
                            return f'<a href="https://finance.yahoo.com/quote/{code}" target="_blank" class="stock-link">{name}</a>'

                    df_filtered['Name'] = df_filtered['Name'].apply(create_link)

                formatters = {}
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df_filtered.columns:
                        formatters[col] = lambda x: '{:,.0f}'.format(x)
                
                if 'Drawdown' in df_filtered.columns:
                    formatters['Drawdown'] = lambda x: '{:.1f}'.format(x * 100)

                # HTML 변환
                tables_html[key] = df_filtered.to_html(
                    classes="table table-hover table-sm table-bordered", 
                    index=False,
                    escape=False,
                    formatters=formatters,                    
                    float_format=lambda x: '{:.1f}'.format(x)
                )
            except Exception as e:
                tables_html[key] = f"<div class='alert alert-danger'>에러: {str(e)}</div>"
                counts[key] = 0
        else:
            tables_html[key] = f"<div class='alert alert-secondary'>데이터 없음</div>"
            counts[key] = 0

    return templates.TemplateResponse("index.html", {
        "request": request,
        "all_columns": ALL_COLUMNS,
        "markets": MARKETS,
        "tables": tables_html,
        "counts": counts,  # 카운트 전달
        "chart_data": json.dumps(chart_data), # JSON 직렬화하여 전달
        "selected_date": target_date,
        "selected_market": market,
        "selected_columns": selected_cols
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)