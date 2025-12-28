from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.io as pio

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_sample_data():
    """가상의 10일치 AAPL 데이터프레임 생성"""
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(9, -1, -1)]
    
    # 220달러 근처에서 랜덤하게 움직이는 데이터 생성
    np.random.seed(42) # 결과값 고정을 위해 설정
    prices = [220 + np.random.uniform(-5, 5) for _ in range(10)]
    
    df = pd.DataFrame({
        "Date": dates,
        "Close": prices
    })
    return df

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    df = get_sample_data()

    fig = px.line(df, x='Date', y='Close', 
                  title='AAPL (Mockup) - Drag to Pan / Wheel to Zoom',
                  markers=True,
                  template="plotly_white")
    
    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        # 드래그했을 때 기본 동작을 '이동(pan)'으로 설정
        dragmode='pan' 
    )

    # 마우스 휠 줌을 활성화하는 config 전달
    plot_div = pio.to_html(
        fig, 
        full_html=False, 
        include_plotlyjs='cdn',
        config={
            'scrollZoom': True,  # 휠로 확대/축소
            'displayModeBar': True, # 우측 상단 도구바 표시
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'] # 불필요한 도구 제거 (선택사항)
        }
    )

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "title": "Interactive AAPL Chart", 
            "sidebar_text": "Click and drag to move the chart!",
            "plot_div": plot_div
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)