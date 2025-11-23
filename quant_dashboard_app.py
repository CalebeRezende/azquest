import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Tenta importar yfinance; se não tiver, instala na hora
try:
    import yfinance as yf
except ModuleNotFoundError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

TRADING_DAYS = 252

# ======================================================
#  UNIVERSO DE INVESTIMENTOS (tipo "lista de produtos")
# ======================================================

INVESTMENT_UNIVERSE: Dict[str, str] = {
    # Brasil
    "PETR4 (Petrobras PN)": "PETR4.SA",
    "VALE3 (Vale ON)": "VALE3.SA",
    "ITUB4 (Itaú Unibanco PN)": "ITUB4.SA",
    "BBDC4 (Bradesco PN)": "BBDC4.SA",
    "BBAS3 (Banco do Brasil ON)": "BBAS3.SA",
    "BOVA11 (ETF Ibovespa)": "BOVA11.SA",
    "IVVB11 (ETF S&P 500 B3)": "IVVB11.SA",

    # EUA (mais “estáveis” no Yahoo)
    "AAPL (Apple)": "AAPL",
    "MSFT (Microsoft)": "MSFT",
    "AMZN (Amazon)": "AMZN",
    "META (Meta)": "META",
    "NVDA (NVIDIA)": "NVDA",
    "TSLA (Tesla)": "TSLA",
    "SPY (ETF S&P 500)": "SPY",
    "QQQ (ETF Nasdaq 100)": "QQQ",
}


# ==========================
#       DATA CLASSES
# ==========================

@dataclass
class AssetStats:
    ticker: str
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


# ==========================
#   FUNÇÕES FINANCEIRAS
# ==========================

def _extract_adj_close_for_single_ticker(
    data: pd.DataFrame | pd.Series,
    ticker: str,
) -> pd.Series | None:
    """
    Extrai a série de 'Adj Close' para UM ticker, independente do formato
    que o yfinance devolveu (Series, DataFrame simples ou MultiIndex).
    Retorna None se não conseguir extrair.
    """
    if data is None or len(data) == 0:
        return None

    # Series
    if isinstance(data, pd.Series):
        if "Adj Close" in data:
            s = data["Adj Close"]
        else:
            return None

    # DataFrame
    else:
        cols = data.columns
        # MultiIndex (ex: ('Adj Close','PETR4.SA'))
        if isinstance(cols, pd.MultiIndex):
            level0 = cols.get_level_values(0)
            level1 = cols.get_level_values(1)

            if "Adj Close" in level0:
                s = data["Adj Close"]
                if isinstance(s, pd.DataFrame):
                    if ticker in s.columns:
                        s = s[ticker]
                    else:
                        s = s.iloc[:, 0]
            elif "Adj Close" in level1:
                s = data.xs("Adj Close", axis=1, level=1)
                if isinstance(s, pd.DataFrame):
                    if ticker in s.columns:
                        s = s[ticker]
                    else:
                        s = s.iloc[:, 0]
            else:
                return None
        else:
            # DataFrame simples, sem MultiIndex
            if "Adj Close" in cols:
                s = data["Adj Close"]
            else:
                return None

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = s.dropna()
    if s.empty:
        return None

    return s


def download_prices(tickers: List[str], years: int = 3) -> pd.DataFrame:
    """
    Baixa preços 'Adj Close' via yfinance para 1 ou vários tickers.
    Faz download de CADA TICKER separadamente, ignora os que falharem
    e continua com o resto.
    """
    end = datetime.today()
    start = end - timedelta(days=365 * years)

    series_map: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            failed.append(t)
            continue

        s = _extract_adj_close_for_single_ticker(data, t)
        if s is None or s.empty:
            failed.append(t)
            continue

        series_map[t] = s

    if failed:
        st.warning(
            "Não foi possível obter dados para: "
            + ", ".join(failed)
            + ". Esses ativos foram ignorados."
        )

    if not series_map:
        raise ValueError("Nenhum dado válido encontrado para os tickers informados.")

    df = pd.DataFrame(series_map)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("Nenhum dado válido após alinhamento e limpeza.")

    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diários."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diários."""
    return prices.pct_change().dropna()


def max_drawdown(returns: pd.Series) -> float:
    """Máximo drawdown a partir de retornos simples (1 + r)."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def annualized_stats(returns: pd.Series, risk_free: float = 0.0) -> Tuple[float, float, float]:
    """Retorno anualizado, volatilidade anualizada e Sharpe anualizado."""
    mean_daily = returns.mean()
    std_daily = returns.std()

    annual_return = float(mean_daily * TRADING_DAYS)
    annual_vol = float(std_daily * math.sqrt(TRADING_DAYS))

    if annual_vol == 0:
        sharpe = 0.0
    else:
        sharpe = (annual_return - risk_free) / annual_vol

    return annual_return, annual_vol, float(sharpe)


def compute_asset_stats(prices: pd.DataFrame, risk_free: float = 0.0) -> List[AssetStats]:
    log_returns = compute_log_returns(prices)
    simple_returns = compute_simple_returns(prices)

    stats: List[AssetStats] = []

    for col in log_returns.columns:
        log_series = log_returns[col].dropna()
        ann_ret, ann_vol, sharpe = annualized_stats(log_series, risk_free)

        simple_series = simple_returns[col].dropna()
        mdd = max_drawdown(simple_series)

        stats.append(
            AssetStats(
                ticker=str(col),
                annual_return=ann_ret,
                annual_vol=ann_vol,
                sharpe=sharpe,
                max_drawdown=mdd,
            )
        )

    return stats


def simulate_investment(prices: pd.DataFrame, invested: float) -> pd.DataFrame:
    """
    Simula quanto cada ativo teria rendido com base no preço inicial e final.
    """
    initial = prices.iloc[0]
    final = prices.iloc[-1]

    returns = (final / initial) - 1
    final_value = invested * (1 + returns)

    df = pd.DataFrame({
        "Retorno (%)": returns * 100,
        "Valor Final (R$)": final_value,
    })

    return df.sort_values("Valor Final (R$)", ascending=False)


# ==========================
#        STREAMLIT UI
# ==========================

def main():
    st.set_page_config(
        page_title="Dashboard Quantitativo",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Dashboard Quantitativo – Estilo Bloomberg")
    st.caption(
        "Escolha uma lista de investimentos, período e valor, e compare desempenho, risco e valor final."
    )

    # --------- CONTROLES (BLOCO TIPO TERMINAL) ----------
    with st.container():
        col_left, col_mid, col_right = st.columns([3, 2, 2])

        with col_left:
            st.subheader("🎯 Seleção de investimentos")

            universe_labels = list(INVESTMENT_UNIVERSE.keys())

            default_selection = [
                "AAPL (Apple)",
                "MSFT (Microsoft)",
                "SPY (ETF S&P 500)",
            ]

            selected_labels = st.multiselect(
                "Escolha os ativos para analisar",
                options=universe_labels,
                default=[l for l in default_selection if l in universe_labels],
                help="Você pode escolher vários ativos (ex: 5, 10, 15) para comparar.",
            )

        with col_mid:
            st.subheader("⏳ Período / Risco")

            years = st.slider(
                "Anos de histórico",
                min_value=1,
                max_value=10,
                value=3,
            )

            risk_free = st.number_input(
                "Taxa livre de risco (ao ano, em %)",
                value=10.0,
                step=0.5,
            ) / 100.0

        with col_right:
            st.subheader("💰 Simulador")

            invested = st.number_input(
                "Valor investido (R$)",
                min_value=100.0,
                value=1000.0,
                step=100.0,
            )

            run_button = st.button("🚀 Rodar análise completa")

    if not selected_labels:
        st.warning("Selecione pelo menos um investimento na lista para começar.")
        return

    tickers = [INVESTMENT_UNIVERSE[label] for label in selected_labels]

    if not run_button:
        st.info("Ajuste os parâmetros acima e clique em **🚀 Rodar análise completa**.")
        return

    # --------- PROCESSAMENTO DE DADOS ----------
    try:
        prices = download_prices(tickers, years=years)
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return

    try:
        stats = compute_asset_stats(prices, risk_free=risk_free)
        sim_df = simulate_investment(prices, invested)
        log_returns = compute_log_returns(prices)
    except Exception as e:
        st.error(f"Erro ao processar dados: {e}")
        return

    # ==========================
    #  BLOCO 1 – MÉTRICAS + TABELA
    # ==========================

    st.markdown("---")
    st.subheader("📊 Visão geral dos ativos")

    df_stats = pd.DataFrame(
        [
            {
                "Ticker": s.ticker,
                "Retorno Anual (%)": s.annual_return * 100,
                "Vol Anual (%)": s.annual_vol * 100,
                "Sharpe": s.sharpe,
                "Max Drawdown (%)": s.max_drawdown * 100,
            }
            for s in stats
        ]
    ).set_index("Ticker")

    # Painel de destaques tipo “Bloomberg”
    col_a, col_b, col_c, col_d = st.columns(4)

    # Escolher o melhor/ pior por alguns critérios
    best_return = df_stats["Retorno Anual (%)"].idxmax()
    best_sharpe = df_stats["Sharpe"].idxmax()
    lowest_dd = df_stats["Max Drawdown (%)"].idxmax()  # menos negativo = melhor

    with col_a:
        st.metric(
            "Melhor retorno anual",
            f"{df_stats.loc[best_return, 'Retorno Anual (%)']:.2f}%",
            help=f"Ticker: {best_return}",
        )
    with col_b:
        st.metric(
            "Melhor Sharpe",
            f"{df_stats.loc[best_sharpe, 'Sharpe']:.2f}",
            help=f"Ticker: {best_sharpe}",
        )
    with col_c:
        st.metric(
            "Menor drawdown (max)",
            f"{df_stats.loc[lowest_dd, 'Max Drawdown (%)']:.2f}%",
            help=f"Ticker: {lowest_dd}",
        )
    with col_d:
        st.metric(
            "Ativos analisados",
            f"{len(df_stats)}",
        )

    st.dataframe(
        df_stats.style.format({
            "Retorno Anual (%)": "{:.2f}",
            "Vol Anual (%)": "{:.2f}",
            "Sharpe": "{:.2f}",
            "Max Drawdown (%)": "{:.2f}",
        }),
        use_container_width=True,
    )

    # ==========================
    #  BLOCO 2 – GRÁFICOS EM DOIS LADOS
    # ==========================

    st.markdown("---")
    st.subheader("📈 Gráficos – Preço, Retorno e Risco")

    col_left_chart, col_right_chart = st.columns(2)

    # --------- ESQUERDA: PREÇOS NORMALIZADOS ---------
    with col_left_chart:
        st.markdown("**Evolução dos preços (normalizados)**")

        norm_prices = prices / prices.iloc[0] * 100  # base 100

        st.line_chart(
            norm_prices,
            height=350,
        )
        st.caption("Base 100 = valor inicial. Estilo terminal: fácil comparar trajetórias.")

    # --------- DIREITA: RETORNO ANUAL X RISCO ---------
    with col_right_chart:
        st.markdown("**Risco x Retorno (annualizado)**")

        df_risk_return = df_stats[["Retorno Anual (%)", "Vol Anual (%)", "Sharpe"]].copy()
        df_risk_return_reset = df_risk_return.reset_index(names="Ticker")

        # Scatter tipo Bloomberg: Vol x Retorno
        import plotly.express as px

        fig_rr = px.scatter(
            df_risk_return_reset,
            x="Vol Anual (%)",
            y="Retorno Anual (%)",
            size="Sharpe",
            color="Ticker",
            hover_data=["Sharpe"],
            height=350,
        )
        fig_rr.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_rr, use_container_width=True)

    # ==========================
    #  BLOCO 3 – SIMULADOR DE INVESTIMENTO
    # ==========================

    st.markdown("---")
    st.subheader(f"💰 Simulador de investimento – R$ {invested:,.2f}")

    col_sim_left, col_sim_right = st.columns([2, 2])

    with col_sim_left:
        st.markdown("**Ranking por valor final**")
        st.dataframe(
            sim_df.style.format({
                "Retorno (%)": "{:.2f}",
                "Valor Final (R$)": "R${:,.2f}",
            }),
            use_container_width=True,
        )

    with col_sim_right:
        st.markdown("**Gráfico – Valor final por ativo**")
        sim_plot = sim_df.copy().reset_index(names="Ticker")
        st.bar_chart(
            sim_plot,
            x="Ticker",
            y="Valor Final (R$)",
            height=350,
        )
        st.caption("Mostra quanto o valor inicial teria virado em cada ativo no período.")

    # ==========================
    #  BLOCO 4 – CORRELAÇÃO (EXTRA)
    # ==========================

    st.markdown("---")
    st.subheader("🔗 Correlação entre ativos (retornos diários)")

    corr = log_returns.corr()
    st.dataframe(
        corr.style.format("{:.2f}"),
        use_container_width=True,
    )
    st.caption("Correlação baseada em retornos logarítmicos diários.")

if __name__ == "__main__":
    main()
