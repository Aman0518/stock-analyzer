# app.py
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List
from io import BytesIO

# -----------------------
# Page config + styles
# -----------------------
st.set_page_config(page_title="StockSage — Polished Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a 0%, #071024 100%); color: #e6eef8; }
    .section-title { color: #00c0ff; font-weight:700; font-size:20px; }
    .card { background: rgba(255,255,255,0.04); padding:12px; border-radius:10px; }
    .muted { color:#bcd2e6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='color:#00c0ff; font-size:60px; font-weight:bold;'>📈 StockSage — Polished Multi-Stock Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="muted" style="font-size:40px;">Enter tickers or company names; pick presets; analyze and download charts.</div>',
    unsafe_allow_html=True
)

st.write("")

# -----------------------
# Friendly name -> ticker mapping (expandable)
# -----------------------
COMPANY_TO_TICKER = {
    "MICROSOFT": "MSFT", "APPLE": "AAPL", "TESLA": "TSLA", "AMAZON": "AMZN",
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "META": "META", "NVIDIA": "NVDA",
    "NETFLIX": "NFLX", "INTEL": "INTC", "AMD": "AMD", "DISNEY": "DIS",
    "IBM": "IBM", "ORACLE": "ORCL", "ADOBE": "ADBE", "WALMART": "WMT",
    "VISA": "V", "MASTERCARD": "MA", "JPMORGAN": "JPM", "BAC": "BAC",
    "COCA-COLA": "KO", "PEPSI": "PEP", "ALIBABA": "BABA"
}
# list of default tickers for quick selection (unique)
PRESET_TICKERS = sorted(set(COMPANY_TO_TICKER.values()))

# -----------------------
# YFinance helpers (cached)
# -----------------------
@st.cache_data(ttl=300)
def fetch_history(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None

def last_close(hist: pd.DataFrame) -> Optional[float]:
    try:
        return float(hist['Close'].iloc[-1])
    except Exception:
        return None

def sma_from_hist(hist: pd.DataFrame, window: int) -> Optional[float]:
    try:
        return float(hist['Close'].rolling(window=window).mean().iloc[-1])
    except Exception:
        return None

def ema_from_hist(hist: pd.DataFrame, window: int) -> Optional[float]:
    try:
        return float(hist['Close'].ewm(span=window, adjust=False).mean().iloc[-1])
    except Exception:
        return None

def rsi_from_hist(hist: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        data = hist['Close']
        delta = data.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=period-1, adjust=False).mean()
        ema_down = down.ewm(com=period-1, adjust=False).mean()
        rs = ema_up / ema_down
        return float((100 - (100 / (1 + rs))).iloc[-1])
    except Exception:
        return None

def fmt(v: Optional[float], p: int = 2) -> str:
    return f"{v:.{p}f}" if v is not None else "N/A"

# -----------------------
# Sidebar: controls
# -----------------------
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    sma_window = st.slider("SMA window (days)", 5, 200, 30)
    ema_window = st.slider("EMA window (days)", 5, 200, 30)
    plot_mode = st.radio("Plot mode", ["Combined comparison (all tickers)", "Individual charts"], index=0)
    st.markdown("---")
    st.markdown("**Quick pick (click to select)**")
    chosen_presets = st.multiselect("Select presets (optional)", PRESET_TICKERS, default=["MSFT", "AAPL", "TSLA"])
    st.markdown("---")
    st.button("Reset UI (reload)", key="reset_ui")  # minor convenience

# -----------------------
# Main input area
# -----------------------
col1, col2 = st.columns([3,1])
with col1:
    text_input =st.markdown(
    '<p style="font-size:20px; font-weight:bold;">Enter company names or tickers (comma or newline separated):</p>',
    unsafe_allow_html=True
)

text_input = st.text_area(
    "",  # leave label empty
    value="Microsoft\nApple",
    height=140
)

with col2:
    st.markdown("**Input tips**")
    st.markdown("- Use tickers (MSFT) or full names (Microsoft).")
    st.markdown("- Separate by commas or new lines.")
    st.markdown("- Use the presets on the left to add common tickers quickly.")
analyze = st.button("Analyze & Plot", type="primary")

# -----------------------
# Utilities: parse and normalize
# -----------------------
def parse_inputs(text: str, presets: List[str]) -> List[str]:
    items = []
    # add presets first
    for p in presets:
        if p and p.strip():
            items.append(p.strip().upper())
    # parse textarea
    if text and text.strip():
        if '\n' in text:
            parts = [p.strip() for p in text.splitlines() if p.strip()]
        else:
            parts = [p.strip() for p in text.split(',') if p.strip()]
        for p in parts:
            if p and p.strip().upper() not in items:
                items.append(p.strip().upper())
    # unique preserve order
    out = []
    seen = set()
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def normalize_to_ticker(token: str) -> str:
    if not token:
        return ""
    up = token.strip().upper()
    if up in COMPANY_TO_TICKER.values():
        return up
    if up in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[up]
    # try title-case names as keys
    if token.title().upper() in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[token.title().upper()]
    # fallback assume ticker
    return up

# -----------------------
# Analysis + results
# -----------------------
if analyze:
    inputs = parse_inputs(text_input, chosen_presets)
    if not inputs:
        st.warning("Please enter at least one ticker or company name (or pick presets).")
    else:
        # resolve inputs -> tickers
        resolved = []
        for inp in inputs:
            ticker = normalize_to_ticker(inp)
            resolved.append((inp, ticker))

        # fetch all histories
        hist_map = {}
        for inp, ticker in resolved:
            hist = fetch_history(ticker, period=period, interval=interval)
            if hist is None:
                st.warning(f"Could not fetch data for '{inp}' → '{ticker}'. Try using ticker symbol (e.g., MSFT).")
            else:
                hist_map[ticker] = hist

        if not hist_map:
            st.error("No valid data fetched. Check tickers and try again.")
        else:
            # top area: combined chart (if selected)
            if plot_mode.startswith("Combined"):
                st.markdown("### 🔀 Combined Comparison")
                fig, ax = plt.subplots(figsize=(10,5))
                for ticker, hist in hist_map.items():
                    ax.plot(hist.index, hist['Close'], label=ticker, linewidth=1.8)
                ax.set_title(f"Close Price Comparison ({period})")
                ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
                ax.grid(alpha=0.25); ax.legend(ncol=2)
                st.pyplot(fig)

                # download combined chart
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                st.download_button("Download combined chart (PNG)", data=buf.getvalue(), file_name="combined_chart.png", mime="image/png")

            # create tabs for each ticker
            tabs = st.tabs(list(hist_map.keys()))
            for idx, ticker in enumerate(list(hist_map.keys())):
                tab = tabs[idx]
                with tab:
                    hist = hist_map[ticker]
                    # top row: metrics in columns
                    col_a, col_b, col_c, col_d = st.columns(4)
                    price = last_close(hist)
                    sma_val = sma_from_hist(hist, sma_window)
                    ema_val = ema_from_hist(hist, ema_window)
                    rsi_val = rsi_from_hist(hist)

                    col_a.metric("Last Close", f"${fmt(price)}")
                    col_b.metric(f"SMA ({sma_window})", f"${fmt(sma_val)}")
                    col_c.metric(f"EMA ({ema_window})", f"${fmt(ema_val)}")
                    col_d.metric("RSI (14)", fmt(rsi_val))

                    # chart + download
                    st.markdown("**Price Chart**")
                    fig2, ax2 = plt.subplots(figsize=(9,4))
                    ax2.plot(hist.index, hist['Close'], label=f"{ticker} Close", linewidth=1.8)
                    # optionally show SMA and EMA lines for visual clarity
                    if len(hist['Close']) >= sma_window:
                        ax2.plot(hist.index, hist['Close'].rolling(window=sma_window).mean(), label=f"SMA({sma_window})", linestyle='--')
                    ax2.plot(hist.index, hist['Close'].ewm(span=ema_window, adjust=False).mean(), label=f"EMA({ema_window})", linestyle=':')
                    ax2.set_title(f"{ticker} Price ({period})"); ax2.set_xlabel("Date"); ax2.set_ylabel("Price (USD)")
                    ax2.grid(alpha=0.2); ax2.legend()
                    st.pyplot(fig2)

                    # download button for single chart
                    buf2 = BytesIO()
                    fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=150)
                    buf2.seek(0)
                    st.download_button(f"Download {ticker} chart (PNG)", data=buf2.getvalue(), file_name=f"{ticker}_chart.png", mime="image/png")

                    # small recent OHLC table
                    st.markdown("**Recent data (last 5 rows)**")
                    try:
                        st.dataframe(hist[['Open','High','Low','Close','Volume']].tail().style.format("{:.2f}"))
                    except Exception:
                        st.dataframe(hist.tail())

            st.success("Analysis complete — use tabs to navigate stocks or download charts.")

# -----------------------
# Footer tips
# -----------------------
st.markdown("---")
st.markdown("**Tips:** Use presets + textarea together. If a full company name isn't recognized, enter the ticker (e.g., MSFT).")
