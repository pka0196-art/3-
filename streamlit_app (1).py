import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pykrx import stock


st.set_page_config(page_title="3일 10% 급등 확률 스크리너", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1400px; padding-top: 1rem; padding-bottom: 2rem;}
    .card {
        padding: 1rem 1rem 0.9rem 1rem;
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 18px;
        margin-bottom: 0.9rem;
        min-height: 345px;
    }
    .rank-badge {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,0.25);
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .title-row {
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .subtle {
        color: #666;
        font-size: 0.92rem;
        margin-bottom: 0.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

THEME_KEYWORDS = [
    "AI", "인공지능", "반도체", "HBM", "로봇", "휴머노이드", "방산", "리튬", "희토류",
    "전기차", "2차전지", "조선", "해운", "원전", "바이오", "제약", "데이터센터",
    "전력", "우주", "스페이스", "유가", "천연가스", "구리", "철강", "정유", "건설",
    "통신", "보안", "드론", "자동차", "플랫폼", "헬스케어"
]

POSITIVE_NEWS_KEYWORDS = [
    "수주", "계약", "실적", "호조", "확대", "신제품", "상용화", "승인", "협력", "증설",
    "공급", "투자", "테마", "급등", "개선", "흑자", "기대", "전망", "출시", "납품"
]


@dataclass
class PickResult:
    code: str
    name: str
    market: str
    current_price: float
    turnover: float
    spike_score: float
    grade: str
    setup: str
    trend: str
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    one_day_return: float
    three_day_return: float
    vol_ratio: float
    atr_pct: float
    reasons: list[str] = field(default_factory=list)
    news_items: list[dict] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None


def format_price(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{float(value):,.0f}"


def format_amount_krw(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    value = float(value)
    if value >= 100_000_000:
        return f"{value / 100_000_000:.1f}억"
    if value >= 10_000:
        return f"{value / 10_000:.1f}만"
    return f"{value:,.0f}"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["고가"] - df["저가"]
    high_close = (df["고가"] - df["종가"].shift(1)).abs()
    low_close = (df["저가"] - df["종가"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA5"] = out["종가"].rolling(5).mean()
    out["SMA10"] = out["종가"].rolling(10).mean()
    out["SMA20"] = out["종가"].rolling(20).mean()
    out["SMA60"] = out["종가"].rolling(60).mean()
    out["EMA12"] = out["종가"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["종가"].ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["RSI14"] = compute_rsi(out["종가"], 14)
    out["ATR14"] = compute_atr(out, 14)
    out["VOL_MA20"] = out["거래량"].rolling(20).mean()
    out["VOL_RATIO"] = out["거래량"] / out["VOL_MA20"].replace(0, np.nan)
    out["HIGH20"] = out["고가"].rolling(20).max()
    out["LOW20"] = out["저가"].rolling(20).min()
    out["RET1"] = out["종가"].pct_change(1) * 100
    out["RET3"] = out["종가"].pct_change(3) * 100
    out["RET5"] = out["종가"].pct_change(5) * 100
    out["TURNOVER_MA20"] = out["거래대금"].rolling(20).mean()
    out["TURNOVER_RATIO"] = out["거래대금"] / out["TURNOVER_MA20"].replace(0, np.nan)
    return out


def compute_grade(score: float) -> str:
    if score >= 82:
        return "A"
    if score >= 70:
        return "B"
    if score >= 58:
        return "C"
    return "D"


def detect_setup(row: pd.Series) -> tuple[str, str]:
    close = row["종가"]
    sma20 = row["SMA20"]
    sma60 = row["SMA60"]
    high20 = row["HIGH20"]
    rsi = row["RSI14"]
    vol_ratio = row["VOL_RATIO"]
    ret3 = row["RET3"]
    atr_pct = (row["ATR14"] / close) * 100 if close else np.nan
    macd = row["MACD"]
    macd_signal = row["MACD_SIGNAL"]

    vals = [sma20, sma60, high20, rsi, vol_ratio, ret3, atr_pct, macd, macd_signal]
    if any(pd.isna(x) for x in vals):
        return "데이터부족", "지표 계산에 필요한 데이터가 부족합니다."

    if close > sma20 > sma60 and close >= high20 * 0.98 and vol_ratio >= 1.6 and macd > macd_signal:
        return "돌파임박형", "20일 고점권 접근 + 거래량 확장 + 정배열"

    if close > sma20 > sma60 and abs(close - sma20) <= row["ATR14"] * 0.9 and 1 <= ret3 <= 7:
        return "눌림후재가속형", "상승 추세 눌림 뒤 재가속 가능 구간"

    if vol_ratio >= 2.2 and 2 <= ret3 <= 12 and 3 <= atr_pct <= 10:
        return "급등초입형", "거래량과 변동성이 동시에 살아난 초기 구간"

    if close < sma20 < sma60:
        return "약세형", "이평 역배열로 급등 확률이 낮은 구간"

    return "관망형", "명확한 급등 전조 패턴은 약합니다."


def calculate_spike_score(row: pd.Series, setup: str):
    score = 0.0
    reasons = []

    close = float(row["종가"])
    sma10 = row["SMA10"]
    sma20 = row["SMA20"]
    sma60 = row["SMA60"]
    rsi = row["RSI14"]
    macd = row["MACD"]
    macd_signal = row["MACD_SIGNAL"]
    vol_ratio = row["VOL_RATIO"]
    turnover_ratio = row["TURNOVER_RATIO"]
    high20 = row["HIGH20"]
    ret1 = row["RET1"]
    ret3 = row["RET3"]
    atr_pct = (row["ATR14"] / close) * 100 if close else 0
    turnover = float(row["거래대금"])

    if close > sma20:
        score += 10
        reasons.append("현재가가 20일선 위")
    if sma20 > sma60:
        score += 12
        reasons.append("20일선 > 60일선")
    if close > sma10:
        score += 6
        reasons.append("10일선 위 단기 모멘텀")
    if macd > macd_signal:
        score += 10
        reasons.append("MACD 골든 방향")
    if 48 <= rsi <= 72:
        score += 10
        reasons.append("RSI 급등 가능 구간")
    elif rsi > 78:
        score -= 8
        reasons.append("RSI 과열 부담")
    if vol_ratio >= 1.5:
        score += 14
        reasons.append("거래량 증가")
    elif vol_ratio < 0.8:
        score -= 5
        reasons.append("거래량 부족")
    if turnover_ratio >= 1.5:
        score += 8
        reasons.append("거래대금 유입")
    if close >= high20 * 0.98:
        score += 12
        reasons.append("20일 고점권 접근")
    if 2 <= ret3 <= 10:
        score += 10
        reasons.append("3일 수익률 초기 확산 구간")
    elif ret3 > 15:
        score -= 6
        reasons.append("단기 과열 가능성")
    if -1 <= ret1 <= 5:
        score += 5
        reasons.append("당일 과열 아닌 출발")
    if 3 <= atr_pct <= 9:
        score += 10
        reasons.append("3일 급등용 변동성 적정")
    elif atr_pct > 12:
        score -= 5
        reasons.append("변동성 과도")
    if turnover >= 5_000_000_000:
        score += 6
        reasons.append("거래대금 충분")

    setup_bonus = {
        "돌파임박형": 16,
        "눌림후재가속형": 13,
        "급등초입형": 18,
        "관망형": 2,
        "데이터부족": -15,
        "약세형": -20,
    }.get(setup, 0)
    score += setup_bonus
    reasons.append(f"패턴 점수: {setup}")

    score = max(0, min(100, round(score, 1)))
    trend = "상승 우위" if close > sma20 > sma60 else "중립 또는 약세"
    return score, reasons, trend, float(ret1 if not pd.isna(ret1) else 0), float(ret3 if not pd.isna(ret3) else 0), float(vol_ratio if not pd.isna(vol_ratio) else 0)


def calculate_trade_prices(row: pd.Series, setup: str, stop_pct: float = 0.06):
    close = float(row["종가"])
    sma5 = float(row["SMA5"]) if not pd.isna(row["SMA5"]) else close
    sma10 = float(row["SMA10"]) if not pd.isna(row["SMA10"]) else close
    high20 = float(row["HIGH20"]) if not pd.isna(row["HIGH20"]) else close
    low20 = float(row["LOW20"]) if not pd.isna(row["LOW20"]) else None

    if setup == "돌파임박형":
        entry = max(high20 * 1.002, close * 0.998)
    elif setup == "눌림후재가속형":
        entry = min(close * 1.002, sma10 * 1.002)
    elif setup == "급등초입형":
        entry = max(sma5, close * 0.995)
    else:
        entry = close

    stop = entry * (1 - stop_pct)
    if low20 and stop > low20:
        stop = min(stop, low20 * 0.997)

    target = max(entry * 1.10, entry + max(entry - stop, entry * 0.03) * 1.8)
    return entry, stop, target


def fetch_google_news(query: str, days: int = 14, max_items: int = 5):
    if not query:
        return []
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {"User-Agent": "Mozilla/5.0"}
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        resp = requests.get(rss_url, timeout=12, headers=headers)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return []

    items = []
    seen = set()
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_raw = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else "출처미상"

        if not title or title in seen:
            continue

        published = None
        if pub_date_raw:
            try:
                published = parsedate_to_datetime(pub_date_raw)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except Exception:
                published = None

        if published and published < cutoff:
            continue

        seen.add(title)
        items.append({
            "title": title,
            "link": link,
            "source": source,
            "published": published.astimezone(timezone.utc).strftime("%Y-%m-%d") if published else "-",
        })
        if len(items) >= max_items:
            break
    return items


def detect_themes(titles):
    joined = " ".join(titles)
    out = []
    for kw in THEME_KEYWORDS:
        if kw.lower() in joined.lower() and kw not in out:
            out.append(kw)
    return out[:6]


def latest_market_date():
    today = datetime.now()
    for i in range(10):
        dt = (today - timedelta(days=i)).strftime("%Y%m%d")
        try:
            test = stock.get_market_ohlcv_by_ticker(dt, market="KOSPI")
            if test is not None and not test.empty:
                return dt
        except Exception:
            pass
    return datetime.now().strftime("%Y%m%d")


@st.cache_data(ttl=900, show_spinner=False)
def load_candidate_universe(market_scope: str, max_candidates: int):
    dt = latest_market_date()
    frames = []

    scopes = []
    if market_scope == "코스피":
        scopes = [("KOSPI", "코스피")]
    elif market_scope == "코스닥":
        scopes = [("KOSDAQ", "코스닥")]
    else:
        scopes = [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]

    for market_code, label in scopes:
        try:
            ohlcv = stock.get_market_ohlcv_by_ticker(dt, market=market_code)
        except Exception:
            ohlcv = pd.DataFrame()
        if ohlcv is None or ohlcv.empty:
            continue
        ohlcv = ohlcv.reset_index()
        first_col = ohlcv.columns[0]
        ohlcv = ohlcv.rename(columns={first_col: "종목코드"})
        ohlcv["시장"] = label
        ohlcv["종목명"] = ohlcv["종목코드"].map(lambda x: stock.get_market_ticker_name(str(x)))
        frames.append(ohlcv)

    if not frames:
        return pd.DataFrame(columns=["종목코드", "종목명", "시장", "거래대금"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["종가"] > 0].copy()
    merged = merged.sort_values("거래대금", ascending=False).head(max_candidates).reset_index(drop=True)
    return merged


def fetch_ohlcv(code: str, lookback_days: int = 280):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    try:
        df = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def analyze_code(code: str, name: str, market: str, stop_pct: float):
    df = fetch_ohlcv(code)
    if df is None or df.empty or len(df) < 80:
        return None

    df = enrich_indicators(df)
    row = df.iloc[-1]

    setup, setup_reason = detect_setup(row)
    score, reasons, trend, ret1, ret3, vol_ratio = calculate_spike_score(row, setup)
    if score < 52 or setup == "약세형":
        return None

    entry, stop, target = calculate_trade_prices(row, setup, stop_pct=stop_pct)
    atr_pct = float((row["ATR14"] / row["종가"]) * 100) if row["종가"] else 0.0

    full_reasons = [setup_reason] + reasons
    full_reasons.append("목표가는 3일 +10% 기준 또는 손익비 기반 중 더 큰 값 적용")

    return PickResult(
        code=code,
        name=name,
        market=market,
        current_price=float(row["종가"]),
        turnover=float(row["거래대금"]),
        spike_score=score,
        grade=compute_grade(score),
        setup=setup,
        trend=trend,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        one_day_return=ret1,
        three_day_return=ret3,
        vol_ratio=vol_ratio,
        atr_pct=atr_pct,
        reasons=full_reasons,
        df=df,
    )


@st.cache_data(ttl=900, show_spinner=False)
def scan_three_day_spike_candidates(market_scope: str, max_candidates: int, stop_pct: float, news_days: int):
    universe = load_candidate_universe(market_scope, max_candidates)
    if universe.empty:
        return []

    results = []
    for _, row in universe.iterrows():
        try:
            result = analyze_code(str(row["종목코드"]), str(row["종목명"]), str(row["시장"]), stop_pct)
            if result is not None:
                results.append(result)
        except Exception:
            continue
        time.sleep(0.01)

    results.sort(key=lambda x: x.spike_score, reverse=True)
    top_results = results[:20]

    for r in top_results[:10]:
        news = fetch_google_news(r.name, days=news_days, max_items=5)
        r.news_items = news
        r.themes = detect_themes([x["title"] for x in news])
        positive_hits = 0
        for item in news:
            title = item["title"]
            if any(k.lower() in title.lower() for k in POSITIVE_NEWS_KEYWORDS):
                positive_hits += 1
        if positive_hits:
            r.spike_score = min(100, round(r.spike_score + min(positive_hits * 2, 6), 1))
            r.reasons.append(f"최근 뉴스 모멘텀 {positive_hits}건 반영")
        if r.themes:
            r.reasons.append("감지 테마: " + ", ".join(r.themes[:4]))

    top_results.sort(key=lambda x: x.spike_score, reverse=True)
    return top_results[:20]


def build_chart(df: pd.DataFrame, name: str):
    last_df = df.tail(90).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=last_df.index,
            open=last_df["시가"],
            high=last_df["고가"],
            low=last_df["저가"],
            close=last_df["종가"],
            name="캔들",
        )
    )
    fig.add_trace(go.Scatter(x=last_df.index, y=last_df["SMA20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=last_df.index, y=last_df["SMA60"], mode="lines", name="SMA60"))
    fig.update_layout(
        title=f"{name} 최근 90거래일 차트",
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def find_stock_code_by_name(name: str):
    dt = latest_market_date()
    for mkt, label in [("KOSPI", "코스피"), ("KOSDAQ", "코스닥")]:
        try:
            tickers = stock.get_market_ticker_list(dt, market=mkt)
        except Exception:
            tickers = []
        for t in tickers:
            nm = stock.get_market_ticker_name(t)
            if nm == name:
                return t, label
    return None, None


def analyze_single_stock(name_or_code: str, stop_pct: float, news_days: int):
    raw = (name_or_code or "").strip()
    if not raw:
        return None

    if raw.isdigit() and len(raw) == 6:
        code = raw
        try:
            name = stock.get_market_ticker_name(code)
        except Exception:
            return None
        market = "코스피" if code.startswith(("0", "1", "2", "3")) else "코스닥"
    else:
        code, market = find_stock_code_by_name(raw)
        name = raw
        if not code:
            return None

    result = analyze_code(code=code, name=name, market=market, stop_pct=stop_pct)
    if result:
        news = fetch_google_news(result.name, days=news_days, max_items=5)
        result.news_items = news
        result.themes = detect_themes([x["title"] for x in news])
        if result.themes:
            result.reasons.append("감지 테마: " + ", ".join(result.themes[:4]))
    return result


st.title("📈 3일 내 10% 급등 확률 스크리너")
st.caption("한국시장 전체 또는 지정 시장을 스캔해, 3일 안에 +10% 급등 가능성이 상대적으로 높은 후보를 점수화해서 보여주는 프로그램입니다.")

with st.expander("사용 전 꼭 읽어주세요", expanded=True):
    st.markdown(
        """
        - 이 앱은 **예언기**가 아니라 **확률형 스크리너**입니다.
        - 즉, **반드시 3일 내 +10%가 나온다**는 뜻이 아니라 **그 가능성이 상대적으로 높은 후보를 선별**합니다.
        - 점수는 차트 패턴, 거래량, 거래대금, 최근 3일 수익률, 변동성, 뉴스 모멘텀을 합쳐 계산합니다.
        - 목표가는 기본적으로 **3일 +10% 목표**를 반영합니다.
        """
    )

tab_auto, tab_search = st.tabs(["시장 스캔 TOP5", "종목 개별 분석"])

with tab_auto:
    st.subheader("시장 스캔 TOP5")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        market_scope = st.selectbox("스캔 시장", ["전체", "코스피", "코스닥"], index=0)
    with c2:
        max_candidates = st.selectbox("사전 후보 개수", [200, 400, 800], index=1)
    with c3:
        stop_label = st.selectbox("손절 비율", ["5%", "6%", "7%", "10%"], index=1)
    with c4:
        news_days = st.selectbox("뉴스 반영 기간", [7, 14, 30], index=1)

    run_scan = st.button("3일 10% 급등 후보 TOP5 스캔", use_container_width=True)

    if run_scan:
        stop_pct = {"5%": 0.05, "6%": 0.06, "7%": 0.07, "10%": 0.10}[stop_label]
        with st.spinner("한국시장을 스캔 중입니다..."):
            results = scan_three_day_spike_candidates(
                market_scope=market_scope,
                max_candidates=int(max_candidates),
                stop_pct=stop_pct,
                news_days=int(news_days),
            )

        if not results:
            st.warning("조건에 맞는 후보를 찾지 못했습니다.")
        else:
            top5 = results[:5]
            st.markdown("### 3일 내 10% 급등 확률형 추천 TOP5")
            cols = st.columns(5)

            for idx, (col, r) in enumerate(zip(cols, top5), start=1):
                with col:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="rank-badge">TOP {idx}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="title-row">{r.name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="subtle">{r.market} | 점수 {r.spike_score} / 100 | 등급 {r.grade}</div>', unsafe_allow_html=True)
                    st.write(f"**현재가:** {format_price(r.current_price)}")
                    st.write(f"**매수 추천가:** {format_price(r.entry_price)}")
                    st.write(f"**손절가:** {format_price(r.stop_price)}")
                    st.write(f"**3일 목표가:** {format_price(r.target_price)}")
                    st.write(f"**패턴:** {r.setup}")
                    st.write(f"**3일 수익률:** {r.three_day_return:.2f}%")
                    st.write(f"**거래량 배수:** {r.vol_ratio:.2f}배")
                    st.write("**추천 근거**")
                    for reason in r.reasons[:4]:
                        st.write(f"- {reason}")
                    if r.news_items:
                        st.write(f"**대표 뉴스:** {r.news_items[0]['title']}")
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("### 스캔 결과 순위표")
            rank_df = pd.DataFrame([
                {
                    "순위": i + 1,
                    "종목명": r.name,
                    "시장": r.market,
                    "종목코드": r.code,
                    "급등점수": r.spike_score,
                    "등급": r.grade,
                    "패턴": r.setup,
                    "현재가": format_price(r.current_price),
                    "매수 추천가": format_price(r.entry_price),
                    "손절가": format_price(r.stop_price),
                    "3일 목표가": format_price(r.target_price),
                    "1일 수익률": round(r.one_day_return, 2),
                    "3일 수익률": round(r.three_day_return, 2),
                    "거래량배수": round(r.vol_ratio, 2),
                    "거래대금": format_amount_krw(r.turnover),
                }
                for i, r in enumerate(results[:20])
            ])
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

            st.markdown("### 상세 보기")
            labels = [f"{i+1}. {r.name} ({r.code})" for i, r in enumerate(results[:10])]
            selected_label = st.selectbox("상세 종목 선택", labels)
            selected_idx = labels.index(selected_label)
            selected = results[selected_idx]

            left, right = st.columns([1.15, 1.0])
            with left:
                st.plotly_chart(build_chart(selected.df, selected.name), use_container_width=True)
            with right:
                st.markdown("#### 상세 근거")
                for reason in selected.reasons:
                    st.write(f"- {reason}")
                st.markdown("#### 관련 뉴스")
                if selected.news_items:
                    for item in selected.news_items:
                        if item["link"]:
                            st.markdown(f"- [{item['title']}]({item['link']})")
                        else:
                            st.write(f"- {item['title']}")
                        st.caption(f"{item['source']} | {item['published']}")
                else:
                    st.write("관련 뉴스가 없습니다.")

with tab_search:
    st.subheader("종목 개별 분석")
    s1, s2, s3 = st.columns(3)
    with s1:
        single_input = st.text_input("종목명 또는 6자리 코드", value="삼성전자")
    with s2:
        single_stop = st.selectbox("손절 비율", ["5%", "6%", "7%", "10%"], index=1, key="single_stop")
    with s3:
        single_news_days = st.selectbox("뉴스 반영 기간", [7, 14, 30], index=1, key="single_news_days")

    run_single = st.button("종목 급등 가능성 분석", use_container_width=True)

    if run_single:
        stop_pct = {"5%": 0.05, "6%": 0.06, "7%": 0.07, "10%": 0.10}[single_stop]
        with st.spinner("종목을 분석 중입니다..."):
            result = analyze_single_stock(single_input, stop_pct=stop_pct, news_days=int(single_news_days))

        if result is None:
            st.error("종목을 찾지 못했거나 데이터가 부족합니다.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("종목명", result.name)
            b.metric("현재가", format_price(result.current_price))
            c.metric("급등점수", f"{result.spike_score} / 100")
            d.metric("등급", result.grade)

            st.markdown(f"### 분석 요약\n**{result.name}** 은(는) 현재 **{result.setup}** 성격이 강하고, 3일 내 +10% 급등 가능성 점수는 **{result.spike_score}점** 입니다.")

            x1, x2, x3, x4 = st.columns(4)
            x1.metric("매수 추천가", format_price(result.entry_price))
            x2.metric("손절가", format_price(result.stop_price))
            x3.metric("3일 목표가", format_price(result.target_price))
            x4.metric("3일 수익률", f"{result.three_day_return:.2f}%")

            left, right = st.columns([1.15, 1.0])
            with left:
                st.plotly_chart(build_chart(result.df, result.name), use_container_width=True)
            with right:
                st.markdown("#### 분석 근거")
                for reason in result.reasons:
                    st.write(f"- {reason}")
                st.markdown("#### 관련 뉴스")
                if result.news_items:
                    for item in result.news_items:
                        if item["link"]:
                            st.markdown(f"- [{item['title']}]({item['link']})")
                        else:
                            st.write(f"- {item['title']}")
                        st.caption(f"{item['source']} | {item['published']}")
                else:
                    st.write("관련 뉴스가 없습니다.")

st.divider()
st.markdown(
    """
    **실행 방법**
    1. `pip install streamlit pykrx pandas numpy plotly requests`
    2. `streamlit run three_day_spike_screener.py`
    """
)
