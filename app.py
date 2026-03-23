import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf

# --- 웹페이지 기본 설정 ---
st.set_page_config(page_title="Deep Search Predictor Pro Max", layout="wide", initial_sidebar_state="expanded")

# --- 1. 바이비트(Bybit) 타임머신 데이터 수집 (미국 서버 차단 우회) ---
def get_bybit_data(symbol, interval, total_candles):
    clean_symbol = symbol.replace("-", "").upper()
    if "USDT" not in clean_symbol: clean_symbol += "USDT"
    
    # 바이비트 분봉 기호 맞춤 변환
    interval_map = {'1m': '1', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '2h': '120', '4h': '240', '1d': 'D'}
    bybit_interval = interval_map[interval]
    
    url = "https://api.bybit.com/v5/market/kline"
    all_data = []
    end_time = None
    
    while len(all_data) < total_candles:
        params = {"category": "linear", "symbol": clean_symbol, "interval": bybit_interval, "limit": 1000}
        if end_time:
            params["end"] = end_time
            
        res = requests.get(url, params=params).json()
        
        # 에러 방어막
        if res.get('retCode') != 0:
            raise ValueError(f"데이터 조회 에러: {res.get('retMsg')}")
            
        kline_list = res['result']['list']
        if not kline_list: break
            
        all_data = all_data + kline_list
        # 바이비트는 최신 데이터부터 주므로, 가장 오래된 시간에서 1밀리초를 빼서 다음 과거를 검색
        end_time = str(int(kline_list[-1][0]) - 1)
        
    # 지정한 봉 개수만큼 자르고, 과거->현재 순서로 뒤집기
    all_data = all_data[:total_candles][::-1]
    
    df = pd.DataFrame(all_data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
    df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
    df.set_index('time', inplace=True)
    return df

def wma_safe(s, p):
    weights = np.arange(1, p + 1)
    w_sum = weights.sum()
    res = pd.Series(0.0, index=s.index)
    for i in range(p):
        res += s.shift(i) * weights[p - 1 - i]
    return res / w_sum

# --- 왼쪽 사이드바 메뉴 ---
st.sidebar.title("⚙️ 예측기 설정")
symbol = st.sidebar.selectbox("종목 설정", ["BTC", "ETH", "SOL", "XRP", "DOGE"], index=0)
interval = st.sidebar.selectbox("분봉 설정", ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=3)
limit_str = st.sidebar.selectbox("과거 탐색 범위", ["1,000봉", "3,000봉", "5,000봉", "10,000봉"], index=1)
total_candles = int(limit_str.replace("봉", "").replace(",", ""))

run_btn = st.sidebar.button("🚀 5대 지표 통합 딥서치 시작", use_container_width=True)

st.title(f"🔥 {symbol} 딥서치 예측기 ({interval})")
st.markdown("과거 데이터를 분석하여 가장 유사한 패턴 5개의 미래 20봉을 시뮬레이션합니다.")

if run_btn:
    with st.spinner(f'5대 지표 기준으로 과거 {total_candles}봉 딥서치 중... (잠시만 기다려주세요)'):
        try:
            # 🚨 데이터 수집 함수 변경 (get_binance_data -> get_bybit_data)
            df = get_bybit_data(symbol, interval, total_candles)
            
            p_len = 60 
            f_len = 20 
            
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
            
            conv = (df_total['high_temp'].rolling(9).max() + df_total['low_temp'].rolling(9).min()) / 2
            base = (df_total['high_temp'].rolling(26).max() + df_total['low_temp'].rolling(26).min()) / 2
            df_total['conv'] = conv
            df_total['base'] = base
            df_total['spanA'] = ((conv + base) / 2).shift(26)
            df_total['spanB'] = ((df_total['high_temp'].rolling(52).max() + df_total['low_temp'].rolling(52).min()) / 2).shift(26)
            
            df_total['ema12'] = df_total['close_temp'].ewm(span=12, adjust=False).mean()
            df_total['ema26'] = df_total['close_temp'].ewm(span=26, adjust=False).mean()
            df_total['hma'] = wma_safe(wma_safe(df_total['close_temp'], 25) * 2 - wma_safe(df_total['close_temp'], 50), 7)
            
            macd = df_total['ema12'] - df_total['ema26']
            macd_sig = macd.ewm(span=9, adjust=False).mean()
            df_total['macd_hist'] = macd - macd_sig
            
            delta = df_total['close_temp'].diff()
            up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            down = (-1 * delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            df_total['rsi'] = 100 - (100 / (1 + up / down))
            
            feat_ema = df_total['ema12'] / df_total['ema26']
            feat_hma = df_total['close_temp'] / df_total['hma']
            feat_ichi = df_total['conv'] / df_total['base']
            feat_macd = df_total['macd_hist'] / df_total['close_temp']
            feat_rsi = df_total['rsi'] / 100.0 
            
            curr_ema = feat_ema.iloc[-f_len-p_len : -f_len].values
            curr_hma = feat_hma.iloc[-f_len-p_len : -f_len].values
            curr_ichi = feat_ichi.iloc[-f_len-p_len : -f_len].values
            curr_macd = feat_macd.iloc[-f_len-p_len : -f_len].values
            curr_rsi = feat_rsi.iloc[-f_len-p_len : -f_len].values
            
            search_end = len(df_total) - f_len - p_len - f_len
            distances = []
            
            for i in range(search_end):
                p_ema = feat_ema.iloc[i : i+p_len].values
                p_hma = feat_hma.iloc[i : i+p_len].values
                p_ichi = feat_ichi.iloc[i : i+p_len].values
                p_macd = feat_macd.iloc[i : i+p_len].values
                p_rsi = feat_rsi.iloc[i : i+p_len].values
                
                if np.isnan(p_ema).any() or np.isnan(p_hma).any() or np.isnan(p_ichi).any() or np.isnan(p_macd).any() or np.isnan(p_rsi).any():
                    continue
                    
                diff_ema = np.mean(np.abs(curr_ema - p_ema))
                diff_hma = np.mean(np.abs(curr_hma - p_hma))
                diff_ichi = np.mean(np.abs(curr_ichi - p_ichi))
                diff_macd = np.mean(np.abs(curr_macd - p_macd))
                diff_rsi = np.mean(np.abs(curr_rsi - p_rsi))
                
                total_diff = (diff_ema + diff_hma + diff_ichi + diff_macd + diff_rsi) / 5
                distances.append((total_diff, i))
                
            distances.sort(key=lambda x: x[0])
            top_5_matches = distances[:5]
            matches = len(top_5_matches) 
            
            up_paths_ohlc = []
            down_paths_ohlc = []
            
            for diff, idx in top_5_matches:
                base_p = df_total['close'].iloc[idx + p_len - 1]
                f_ohlc = df_total[['open', 'high', 'low', 'close']].iloc[idx+p_len : idx+p_len+f_len].values / base_p
                
                if f_ohlc[-1, 3] > 1.0: up_paths_ohlc.append(f_ohlc)
                else: down_paths_ohlc.append(f_ohlc)

            for side in ['up', 'down']:
                for col in ['_open', '_high', '_low', '_close']:
                    df_total[f'{side}{col}'] = np.nan

            if len(up_paths_ohlc) > 0:
                pred_up = np.mean(up_paths_ohlc, axis=0) * curr_close
                df_total.loc[f_idx, ['up_open', 'up_high', 'up_low', 'up_close']] = pred_up

            if len(down_paths_ohlc) > 0:
                pred_down = np.mean(down_paths_ohlc, axis=0) * curr_close
                df_total.loc[f_idx, ['down_open', 'down_high', 'down_low', 'down_close']] = pred_down

            if matches > 0:
                for col in ['conv', 'base', 'ema12', 'ema26', 'hma', 'macd_hist', 'rsi']:
                    df_total.loc[f_idx, col] = df_total[col].iloc[-f_len-1]

            plot_df = df_total.iloc[-160:].copy()
            ohlcv_df = plot_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            y1_cloud = plot_df['spanA'].astype(float).bfill().ffill().values
            y2_cloud = plot_df['spanB'].astype(float).bfill().ffill().values
            
            cloud_mask = np.ones(len(plot_df), dtype=bool)
            cloud_mask[-f_len:] = False
            
            macd_colors = ['#26a69a' if val > 0 else '#ef5350' for val in plot_df['macd_hist'].fillna(0)]

            apds = [
                mpf.make_addplot(plot_df['conv'].astype(float).values, color='#2962FF', width=1, panel=0),
                mpf.make_addplot(plot_df['base'].astype(float).values, color='#B71C1C', width=1, panel=0),
                mpf.make_addplot(plot_df['ema12'].astype(float).values, color='#6f92ce', width=1.5, panel=0),
                mpf.make_addplot(plot_df['ema26'].astype(float).values, color='#e08937', width=1.5, panel=0),
                mpf.make_addplot(plot_df['hma'].astype(float).values, color='#ab47bc', width=2, panel=0),
                mpf.make_addplot(plot_df['macd_hist'].astype(float).values, type='bar', color=macd_colors, panel=1, ylabel='MACD'),
                mpf.make_addplot(plot_df['rsi'].astype(float).values, color='#ab47bc', panel=2, ylabel='RSI'),
                mpf.make_addplot([70]*len(plot_df), color='#555555', linestyle='--', panel=2),
                mpf.make_addplot([30]*len(plot_df), color='#555555', linestyle='--', panel=2),
            ]
            
            mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in', ohlc='in')
            dark_style = mpf.make_mpf_style(
                marketcolors=mc, facecolor='#121212', figcolor='#121212', gridcolor='#333333',
                rc={'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
            )
            
            fig, axlist = mpf.plot(
                ohlcv_df, type='candle', style=dark_style, addplot=apds,
                returnfig=True, volume=False, figratio=(16, 10), tight_layout=True,
                panel_ratios=(4, 1.5, 1.5),
                fill_between=dict(y1=y1_cloud, y2=y2_cloud, where=cloud_mask, alpha=0.2, color='#757575')
            )
            
            axlist[0].axvspan(len(ohlcv_df)-f_len-0.5, len(ohlcv_df)-0.5, color='#FFD700', alpha=0.08)

            if len(up_paths_ohlc) > 0:
                for i in range(f_len):
                    idx = len(plot_df) - f_len + i
                    o, h, l, c = plot_df['up_open'].iloc[idx], plot_df['up_high'].iloc[idx], plot_df['up_low'].iloc[idx], plot_df['up_close'].iloc[idx]
                    if pd.isna(o): continue
                    color = 'dodgerblue' if c >= o else '#64b5f6'
                    body_min, body_max = min(o, c), max(o, c)
                    if body_min == body_max: body_max += curr_close * 0.00001
                    axlist[0].vlines(idx - 0.2, l, h, color=color, linewidth=1)
                    axlist[0].vlines(idx - 0.2, body_min, body_max, color=color, linewidth=4)
                axlist[0].plot([], [], color='dodgerblue', label=f'UP ({len(up_paths_ohlc)})', linewidth=4)

            if len(down_paths_ohlc) > 0:
                for i in range(f_len):
                    idx = len(plot_df) - f_len + i
                    o, h, l, c = plot_df['down_open'].iloc[idx], plot_df['down_high'].iloc[idx], plot_df['down_low'].iloc[idx], plot_df['down_close'].iloc[idx]
                    if pd.isna(o): continue
                    color = '#7e57c2' if c >= o else '#b39ddb'
                    body_min, body_max = min(o, c), max(o, c)
                    if body_min == body_max: body_max += curr_close * 0.00001
                    axlist[0].vlines(idx + 0.2, l, h, color=color, linewidth=1)
                    axlist[0].vlines(idx + 0.2, body_min, body_max, color=color, linewidth=4)
                axlist[0].plot([], [], color='#7e57c2', label=f'DOWN ({len(down_paths_ohlc)})', linewidth=4)

            if matches > 0: 
                axlist[0].legend(loc='upper left', fontsize=10, facecolor='#1e1e1e', edgecolor='#333333', labelcolor='white')

            up_moves, down_moves = len(up_paths_ohlc), len(down_paths_ohlc)
            up_prob = (up_moves / matches) * 100 if matches > 0 else 0
            down_prob = (down_moves / matches) * 100 if matches > 0 else 0
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.pyplot(fig) 
                
            with col2:
                st.subheader("💡 예측 결과")
                st.write(f"과거 **{total_candles}봉** 속에서 5대 지표가 가장 똑같은 **Top {matches} 패턴**을 찾아냈습니다.")
                st.metric(label="📈 20봉 뒤 상승 확률", value=f"{up_prob:.0f}%", delta=f"{up_moves}건 매칭")
                st.metric(label="📉 20봉 뒤 하락 확률", value=f"{down_prob:.0f}%", delta=f"-{down_moves}건 매칭")
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
