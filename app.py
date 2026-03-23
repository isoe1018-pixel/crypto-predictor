import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
import time

# --- 웹페이지 기본 설정 (다크모드 강제 적용) ---
st.set_page_config(page_title="Deep Search Predictor Pro Max", layout="wide", initial_sidebar_state="expanded")

# --- 1. 바이낸스 데이터 수집 (미국 IP 방어용 프록시 지원망) ---
def get_binance_data(symbol, interval, total_candles):
    clean_symbol = symbol.replace("-", "").upper()
    if "USDT" not in clean_symbol: clean_symbol += "USDT"
    
    url = "https://data-api.binance.vision/api/v3/klines"
    all_data = []
    end_time = None
    
    while len(all_data) < total_candles:
        params = {"symbol": clean_symbol, "interval": interval, "limit": 1000}
        if end_time:
            params["endTime"] = end_time
            
        try:
            response = requests.get(url, params=params)
            res = response.json()
        except:
            raise ValueError("데이터 서버 연결 실패. 잠시 후 다시 시도해 주세요.")
            
        if not res or not isinstance(res, list): break
            
        all_data = res + all_data
        end_time = res[0][0] - 1
        
    all_data = all_data[-total_candles:]
    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'c_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ign'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    # 🚨 한국 시간(KST) 적용: UTC + 9시간
    df['time'] = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=9)
    df.set_index('time', inplace=True)
    return df

def wma_safe(s, p):
    weights = np.arange(1, p + 1)
    w_sum = weights.sum()
    res = pd.Series(0.0, index=s.index)
    for i in range(p):
        res += s.shift(i) * weights[p - 1 - i]
    return res / w_sum

# --- 사이드바 설정 ---
st.sidebar.title("⚙️ 예측기 설정")
symbol = st.sidebar.selectbox("종목 설정", ["BTC", "ETH", "SOL", "XRP", "DOGE"], index=0)
interval = st.sidebar.selectbox("분봉 설정", ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=3)
limit_str = st.sidebar.selectbox("과거 탐색 범위", ["1,000봉", "3,000봉", "5,000봉", "10,000봉"], index=1)
total_candles = int(limit_str.replace("봉", "").replace(",", ""))

# 🚨 자동 갱신 선택 추가
refresh_option = st.sidebar.selectbox("🔄 자동 갱신 간격", ["사용 안 함", "5분", "10분", "30분"], index=0)

run_btn = st.sidebar.button("🚀 딥서치 시작 / 새로고침", use_container_width=True)

# 🚨 자동 갱신 로직 (Streamlit 타이머)
if refresh_option != "사용 안 함":
    mins = int(refresh_option.replace("분", ""))
    st.info(f"✅ {mins}분마다 자동으로 차트를 갱신합니다.")
    # 지정된 초(seconds)마다 스크립트 재실행
    time.sleep(mins * 60)
    st.rerun()

st.title(f"🔥 {symbol} 딥서치 예측기 ({interval})")

# --- 로직 실행 ---
if run_btn or refresh_option != "사용 안 함":
    with st.spinner('데이터 분석 중...'):
        df = get_binance_data(symbol, interval, total_candles)
        
        p_len, f_len = 60, 20
        last_t = df.index[-1]
        t_gap = df.index[-1] - df.index[-2]
        f_idx = [last_t + t_gap * (i+1) for i in range(f_len)]
        f_df = pd.DataFrame(index=f_idx, columns=df.columns)
        
        df_total = pd.concat([df, f_df])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_total[col] = pd.to_numeric(df_total[col], errors='coerce').astype(float)
        
        curr_close = df['close'].iloc[-1]
        df_total['close_temp'] = df_total['close'].fillna(curr_close).astype(float)
        df_total['high_temp'] = df_total['high'].fillna(curr_close).astype(float)
        df_total['low_temp'] = df_total['low'].fillna(curr_close).astype(float)
        
        # 5대 지표 로직
        conv = (df_total['high_temp'].rolling(9).max() + df_total['low_temp'].rolling(9).min()) / 2
        base = (df_total['high_temp'].rolling(26).max() + df_total['low_temp'].rolling(26).min()) / 2
        df_total['conv'], df_total['base'] = conv, base
        df_total['spanA'] = ((conv + base) / 2).shift(26)
        df_total['spanB'] = ((df_total['high_temp'].rolling(52).max() + df_total['low_temp'].rolling(52).min()) / 2).shift(26)
        df_total['ema12'] = df_total['close_temp'].ewm(span=12, adjust=False).mean()
        df_total['ema26'] = df_total['close_temp'].ewm(span=26, adjust=False).mean()
        df_total['hma'] = wma_safe(wma_safe(df_total['close_temp'], 25) * 2 - wma_safe(df_total['close_temp'], 50), 7)
        df_total['macd_hist'] = (df_total['ema12'] - df_total['ema26']) - (df_total['ema12'] - df_total['ema26']).ewm(span=9, adjust=False).mean()
        delta = df_total['close_temp'].diff()
        up_r = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        down_r = (-1 * delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df_total['rsi'] = 100 - (100 / (1 + up_r / down_r))
        
        # 딥서치 매칭 로직 (동일)
        feat_ema = df_total['ema12'] / df_total['ema26']
        feat_hma = df_total['close_temp'] / df_total['hma']
        feat_ichi = df_total['conv'] / df_total['base']
        feat_macd = df_total['macd_hist'] / df_total['close_temp']
        feat_rsi = df_total['rsi'] / 100.0
        
        curr_p = [feat_ema.iloc[-f_len-p_len:-f_len].values, feat_hma.iloc[-f_len-p_len:-f_len].values, 
                  feat_ichi.iloc[-f_len-p_len:-f_len].values, feat_macd.iloc[-f_len-p_len:-f_len].values, feat_rsi.iloc[-f_len-p_len:-f_len].values]
        
        distances = []
        search_end = len(df_total) - f_len - p_len - f_len
        for i in range(search_end):
            p_feat = [feat_ema.iloc[i:i+p_len].values, feat_hma.iloc[i:i+p_len].values, 
                      feat_ichi.iloc[i:i+p_len].values, feat_macd.iloc[i:i+p_len].values, feat_rsi.iloc[i:i+p_len].values]
            if any(np.isnan(f).any() for f in p_feat): continue
            d = np.mean([np.mean(np.abs(curr_p[j] - p_feat[j])) for j in range(5)])
            distances.append((d, i))
        
        distances.sort(key=lambda x: x[0])
        top_5 = distances[:5]
        up_paths, down_paths = [], []
        
        for d, idx in top_5:
            f_ohlc = df_total[['open', 'high', 'low', 'close']].iloc[idx+p_len : idx+p_len+f_len].values / df_total['close'].iloc[idx+p_len-1]
            if f_ohlc[-1, 3] > 1.0: up_paths.append(f_ohlc)
            else: down_paths.append(f_ohlc)

        for side, paths in [('up', up_paths), ('down', down_paths)]:
            if paths:
                pred = np.mean(paths, axis=0) * curr_close
                df_total.loc[f_idx, [f'{side}_open', f'{side}_high', f'{side}_low', f'{side}_close']] = pred

        # 시각화 데이터 준비
        plot_df = df_total.iloc[-160:].copy()
        
        # 🚨 다크 테마 및 Y축 확장 로직
        mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in', ohlc='in')
        dark_s = mpf.make_mpf_style(marketcolors=mc, facecolor='#121212', figcolor='#121212', gridcolor='#333333', rc={'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})
        
        macd_colors = ['#26a69a' if v > 0 else '#ef5350' for v in plot_df['macd_hist'].fillna(0)]
        apds = [
            mpf.make_addplot(plot_df['conv'], color='#2962FF', width=1, panel=0),
            mpf.make_addplot(plot_df['base'], color='#B71C1C', width=1, panel=0),
            mpf.make_addplot(plot_df['ema12'], color='#6f92ce', width=1.2, panel=0),
            mpf.make_addplot(plot_df['ema26'], color='#e08937', width=1.2, panel=0),
            mpf.make_addplot(plot_df['hma'], color='#ab47bc', width=1.5, panel=0),
            mpf.make_addplot(plot_df['macd_hist'], type='bar', color=macd_colors, panel=1),
            mpf.make_addplot(plot_df['rsi'], color='#ab47bc', panel=2),
            mpf.make_addplot([70]*len(plot_df), color='#555555', linestyle='--', panel=2),
            mpf.make_addplot([30]*len(plot_df), color='#555555', linestyle='--', panel=2)
        ]

        fig, axlist = mpf.plot(plot_df[['open','high','low','close','volume']], type='candle', style=dark_s, addplot=apds, returnfig=True, figratio=(16,9), panel_ratios=(4,1.2,1.2), tight_layout=True, fill_between=dict(y1=plot_df['spanA'].values, y2=plot_df['spanB'].values, where=np.array([i < (len(plot_df)-f_len) for i in range(len(plot_df))]), alpha=0.15, color='#757575'))
        
        # 미래 구역 표시 및 예측 캔들
        axlist[0].axvspan(len(plot_df)-f_len-0.5, len(plot_df)-0.5, color='#FFD700', alpha=0.05)
        
        for side, color, offset in [('up', 'dodgerblue', -0.2), ('down', '#7e57c2', 0.2)]:
            if f'{side}_open' in plot_df.columns:
                for i in range(f_len):
                    idx = len(plot_df) - f_len + i
                    row = plot_df.iloc[idx]
                    if not pd.isna(row[f'{side}_open']):
                        axlist[0].vlines(idx + offset, row[f'{side}_low'], row[f'{side}_high'], color=color, linewidth=1)
                        axlist[0].vlines(idx + offset, min(row[f'{side}_open'], row[f'{side}_close']), max(row[f'{side}_open'], row[f'{side}_close']), color=color, linewidth=4)

        # 🚨 Y축 자동 확장 적용
        all_h = plot_df[['high', 'up_high', 'down_high']].max().max()
        all_l = plot_df[['low', 'up_low', 'down_low']].min().min()
        padding = (all_h - all_l) * 0.05
        axlist[0].set_ylim(all_l - padding, all_h + padding)

        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1: st.metric("📈 상승 확률", f"{(len(up_paths)/5)*100:.0f}%", f"{len(up_paths)}건")
        with col2: st.metric("📉 하락 확률", f"{(len(down_paths)/5)*100:.0f}%", f"{len(down_paths)}건")
        st.caption(f"최종 갱신 (한국 시간): {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
