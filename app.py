import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
import time

# --- 1. 웹페이지 다크모드 설정 ---
st.set_page_config(page_title="Deep Search Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stMetric { background-color: #1C2128; border: 1px solid #30363D; padding: 15px; border-radius: 10px; }
    h1, h2, h3, p { color: #E6EDF3 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 데이터 수집 (글로벌 우회망) ---
def get_binance_data(symbol, interval, total_candles):
    clean_symbol = symbol.replace("-", "").upper() + "USDT"
    url = "https://data-api.binance.vision/api/v3/klines"
    all_data = []
    end_time = None
    
    while len(all_data) < total_candles:
        params = {"symbol": clean_symbol, "interval": interval, "limit": 1000}
        if end_time: params["endTime"] = end_time
        res = requests.get(url, params=params).json()
        if not res or not isinstance(res, list): break
        all_data = res + all_data
        end_time = res[0][0] - 1
        
    df = pd.DataFrame(all_data[-total_candles:], columns=['time', 'open', 'high', 'low', 'close', 'volume', 'c_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ign'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    df['time'] = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=9)
    df.set_index('time', inplace=True)
    return df

def wma_safe(s, p):
    weights = np.arange(1, p + 1)
    return s.rolling(p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# --- 3. 사이드바 제어판 ---
st.sidebar.title("⚙️ 예측기 설정")
symbol = st.sidebar.selectbox("종목", ["BTC", "ETH", "SOL", "XRP", "DOGE"], index=0)
interval = st.sidebar.selectbox("분봉", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=3)
limit_val = st.sidebar.selectbox("과거 탐색 범위", ["1,000봉", "3,000봉", "5,000봉"], index=1)
total_candles = int(limit_val.replace("봉", "").replace(",", ""))

refresh_opt = st.sidebar.selectbox("🔄 자동 갱신 간격", ["사용 안 함", "5분", "10분", "30분"], index=0)
run_btn = st.sidebar.button("🚀 즉시 분석 시작", use_container_width=True)

# --- 4. 메인 로직 ---
st.title(f"🔥 {symbol} 딥서치 예측기 ({interval})")

if run_btn or refresh_opt != "사용 안 함":
    with st.spinner('패턴 분석 중...'):
        try:
            df = get_binance_data(symbol, interval, total_candles)
            p_len, f_len = 60, 20
            
            last_t = df.index[-1]
            t_gap = df.index[-1] - df.index[-2]
            f_idx = [last_t + t_gap * (i+1) for i in range(f_len)]
            f_df = pd.DataFrame(index=f_idx, columns=df.columns)
            
            df_total = pd.concat([df, f_df])
            
            # 🚨 [해결] 예측 컬럼들을 미리 NaN으로 생성 (KeyError 방지)
            for side in ['up', 'down']:
                for col in ['open', 'high', 'low', 'close']:
                    df_total[f'{side}_{col}'] = np.nan
            
            curr_close = df['close'].iloc[-1]
            df_total['close_temp'] = df_total['close'].fillna(curr_close).astype(float)
            df_total['high_temp'] = df_total['high'].fillna(curr_close).astype(float)
            df_total['low_temp'] = df_total['low'].fillna(curr_close).astype(float)
            
            # 지표 계산
            df_total['ema12'] = df_total['close_temp'].ewm(span=12, adjust=False).mean()
            df_total['ema26'] = df_total['close_temp'].ewm(span=26, adjust=False).mean()
            df_total['hma'] = wma_safe(wma_safe(df_total['close_temp'], 25) * 2 - wma_safe(df_total['close_temp'], 50), 7)
            df_total['macd_hist'] = (df_total['ema12'] - df_total['ema26']) - (df_total['ema12'] - df_total['ema26']).ewm(span=9, adjust=False).mean()
            delta = df_total['close_temp'].diff()
            up_r = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            down_r = (-1 * delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            df_total['rsi'] = 100 - (100 / (1 + up_r / down_r))
            
            conv = (df_total['high_temp'].rolling(9).max() + df_total['low_temp'].rolling(9).min()) / 2
            base = (df_total['high_temp'].rolling(26).max() + df_total['low_temp'].rolling(26).min()) / 2
            df_total['spanA'] = ((conv + base) / 2).shift(26)
            df_total['spanB'] = ((df_total['high_temp'].rolling(52).max() + df_total['low_temp'].rolling(52).min()) / 2).shift(26)

            # 패턴 매칭
            feat = (df_total['ema12']/df_total['ema26']).values
            curr_v = feat[-f_len-p_len : -f_len]
            distances = []
            for i in range(len(feat) - f_len - p_len - f_len):
                past_v = feat[i : i+p_len]
                if not np.isnan(past_v).any():
                    distances.append((np.mean(np.abs(curr_v - past_v)), i))
            
            distances.sort(key=lambda x: x[0])
            top_5 = distances[:5]
            up_paths, down_paths = [], []
            
            for _, idx in top_5:
                f_ohlc = df_total[['open', 'high', 'low', 'close']].iloc[idx+p_len : idx+p_len+f_len].values / df_total['close'].iloc[idx+p_len-1]
                if f_ohlc[-1, 3] > 1.0: up_paths.append(f_ohlc)
                else: down_paths.append(f_ohlc)

            for side, paths in [('up', up_paths), ('down', down_paths)]:
                if paths:
                    df_total.loc[f_idx, [f'{side}_open', f'{side}_high', f'{side}_low', f'{side}_close']] = np.mean(paths, axis=0) * curr_close

            # 시각화 데이터
            plot_df = df_total.iloc[-160:].copy()
            mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
            dark_s = mpf.make_mpf_style(marketcolors=mc, facecolor='#0E1117', figcolor='#0E1117', gridcolor='#30363D', rc={'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
            
            apds = [
                mpf.make_addplot(plot_df['ema12'], color='#6f92ce', width=1, panel=0),
                mpf.make_addplot(plot_df['ema26'], color='#e08937', width=1, panel=0),
                mpf.make_addplot(plot_df['hma'], color='#ab47bc', width=1.5, panel=0),
                mpf.make_addplot(plot_df['macd_hist'], type='bar', color=['#26a69a' if v > 0 else '#ef5350' for v in plot_df['macd_hist'].fillna(0)], panel=1),
                mpf.make_addplot(plot_df['rsi'], color='#ab47bc', panel=2)
            ]

            fig, axlist = mpf.plot(plot_df[['open','high','low','close','volume']], type='candle', style=dark_s, addplot=apds, returnfig=True, figratio=(16,9), panel_ratios=(4,1.2,1.2), tight_layout=True,
                                   fill_between=dict(y1=plot_df['spanA'].values, y2=plot_df['spanB'].values, where=np.array([i < (len(plot_df)-f_len) for i in range(len(plot_df))]), alpha=0.1, color='#757575'))
            
            axlist[0].axvspan(len(plot_df)-f_len-0.5, len(plot_df)-0.5, color='#FFD700', alpha=0.03)
            
            for side, color, offset in [('up', '#00BFFF', -0.2), ('down', '#9370DB', 0.2)]:
                for i in range(f_len):
                    idx = len(plot_df) - f_len + i
                    row = plot_df.iloc[idx]
                    if not pd.isna(row[f'{side}_open']):
                        axlist[0].vlines(idx + offset, row[f'{side}_low'], row[f'{side}_high'], color=color, linewidth=1)
                        axlist[0].vlines(idx + offset, min(row[f'{side}_open'], row[f'{side}_close']), max(row[f'{side}_open'], row[f'{side}_close']), color=color, linewidth=4)

            # 🚨 [수정] Y축 자동 확장 로직 (컬럼 존재 여부 체크)
            check_cols_h = [c for c in ['high', 'up_high', 'down_high'] if c in plot_df.columns]
            check_cols_l = [c for c in ['low', 'up_low', 'down_low'] if c in plot_df.columns]
            
            all_h = plot_df[check_cols_h].max().max()
            all_l = plot_df[check_cols_l].min().min()
            pad = (all_h - all_l) * 0.1
            axlist[0].set_ylim(all_l - pad, all_h + pad)

            st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            c1.metric("📈 상승 확률", f"{(len(up_paths)/5)*100:.0f}%", f"{len(up_paths)}건")
            c2.metric("📉 하락 확률", f"{(len(down_paths)/5)*100:.0f}%", f"{len(down_paths)}건")
            st.caption(f"최종 갱신 (한국시간): {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error(f"오류 발생: {e}")

# --- 5. 자동 갱신 타이머 ---
if refresh_opt != "사용 안 함":
    mins = int(refresh_opt.replace("분", ""))
    time.sleep(mins * 60)
    st.rerun()
