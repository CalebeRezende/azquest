from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Garantir que yfinance esteja instalado mesmo se o ambiente ignorar requirements.txt
try:
    import yfinance as yf
except ModuleNotFoundError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

TRADING_DAYS = 252

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


@dataclass
class PortfolioStats:
    weights: Dict[str, float]
    annual_return: float
    annual_vol: float
    sharpe: float


# ==========================
#   FUNÇÕES FINANCEIRAS
# ==========================

def download_prices(tickers: List[str], years: int = 3) -> pd.DataFrame:
    """
    Baixa preços 'Adj Close' via yfinance para 1 ou vários tickers,
    tratando Series, DataFrame simples e MultiIndex.
    """
    end = datetime.today()
    start = end - timedelta(days=365 * years)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if data is None or len(data) == 0:
        raise ValueError("Nenhum dado retornado pelo Yahoo Finance.")

    # Caso Series (situação mais rara)
    if isinstance(data, pd.Series):
        if "Adj Close" in data:
            df = data["Adj Close"].to_frame()
        else:
            raise ValueError(f"Nenhum 'Adj Close' retornado para {tickers}.")
    else:
        cols = data.columns

        # MultiIndex (mais comum com vários tickers)
        if isinstance(cols, pd.MultiIndex):
            level0 = cols.get_level_values(0)
            level1 = cols.get_level_values(1)

            if "Adj Close" in level0:
                df = data["Adj Close"]
            elif "Adj Close" in level1:
                df = data.xs("Adj Close", axis=1, level=1)
            else:
                raise ValueError("MultiIndex sem nível 'Adj Close'.")
        else:
            # DataFrame simples (1 ticker só)
            if "Adj Close" in cols:
                df = data["Adj Close"].to_frame()
            else:
                raise ValueError("DataFrame sem coluna 'Adj Close'.")

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("Nenhum dado válido após limpeza.")

    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diários."""
    return np.log(prices / prices.shift(1)).dropna()


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
    returns_df = compute_returns(prices)
    stats: List[AssetStats] = []

    for col in returns_df.columns:
        series = returns_df[col].dropna()
        ann_ret, ann_vol, sharpe = annualized_stats(series, risk_free)
        # para drawdown, usa retorno simples aproximado
        simple_ret = prices[col].pct_change().dropna()
        mdd = max_drawdown(simple_ret)

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


def random_portfolios(
    returns_df: pd.DataFrame,
    n_portfolios: int = 2000,
    risk_free: float = 0.0,
) -> List[PortfolioStats]:
    mean_daily = returns_df.mean()
    cov_matrix = returns_df.cov()
    tickers = list(returns_df.columns)

    portfolios: List[PortfolioStats] = []

    for _ in range(n_portfolios):
        weights = np.random.random(len(tickers))
        weights /= weights.sum()

        port_daily_return = float(np.dot(weights, mean_daily))
        port_daily_vol = float(
            math.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        )

        ann_return = port_daily_return * TRADING_DAYS
        ann_vol = port_daily_vol * math.sqrt(TRADING_DAYS)

        if ann_vol == 0:
            sharpe = 0.0
        else:
            sharpe = (annual_return - risk_free) / ann_vol

        portfolios.append(
            PortfolioStats(
                weights={t: float(w) for t, w in zip(tickers, weights)},
                annual_return=float(ann_return),
                annual_vol=float(ann_vol),
                sharpe=float(sharpe),
            )
        )

    return portfolios


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
        page_title="Mini Dashboard Quantitativo",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Mini Dashboard Quantitativo em Python")
    st.caption("Análise de ativos, simulação de portfólios e simulador de investimento.")

    col1, col2, col3 = st.columns([3, 1.5, 1.5])

    with col1:
        tickers_str = st.text_input(
            "Tickers (separados por vírgula)",
            value="PETR4.SA, VALE3.SA, ITUB4.SA",
            help="Use o formato do Yahoo Finance (ex: PETR4.SA, VALE3.SA, ITUB4.SA, AAPL, MSFT).",
        )

    with col2:
        years = st.slider(
            "Anos de histórico",
            min_value=1,
            max_value=10,
            value=3,
        )

    with col3:
        risk_free = st.number_input(
            "Taxa livre de risco (ao ano, em %)",
            value=10.0,
            step=0.5,
        ) / 100.0

    analysis_type = st.radio(
        "Tipo de análise",
        options=[
            "Estatísticas por ativo",
            "Simulação de portfólio (Monte Carlo)",
            "Simulador de investimento (quanto rende?)",
        ],
        horizontal=True,
    )

    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) == 0:
        st.warning("Informe pelo menos um ticker para começar.")
        return

    # ============= ESTATÍSTICAS POR ATIVO =============
    if analysis_type == "Estatísticas por ativo":
        if st.button("🚀 Rodar análise de estatísticas", key="btn_stats"):
            try:
                prices = download_prices(tickers, years=years)
                stats = compute_asset_stats(prices, risk_free=risk_free)
            except Exception as e:
                st.error(f"Erro ao baixar dados ou calcular estatísticas: {e}")
                return

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

            st.subheader("📊 Estatísticas por ativo")
            st.dataframe(df_stats.style.format({
                "Retorno Anual (%)": "{:.2f}",
                "Vol Anual (%)": "{:.2f}",
                "Sharpe": "{:.2f}",
                "Max Drawdown (%)": "{:.2f}",
            }))

    # ============= MONTE CARLO DE PORTFÓLIOS ============
    elif analysis_type == "Simulação de portfólio (Monte Carlo)":
        n_portfolios = st.slider(
            "Número de portfólios a simular",
            min_value=500,
            max_value=10000,
            value=3000,
            step=500,
        )

        if st.button("🎲 Rodar simulação de portfólios", key="btn_mc"):
            try:
                prices = download_prices(tickers, years=years)
                returns_df = compute_returns(prices)
                portfolios = random_portfolios(
                    returns_df,
                    n_portfolios=n_portfolios,
                    risk_free=risk_free,
                )
            except Exception as e:
                st.error(f"Erro ao rodar simulação: {e}")
                return

            portfolios_sorted = sorted(portfolios, key=lambda p: p.sharpe, reverse=True)
            top_ports = portfolios_sorted[:5]

            df_ports = pd.DataFrame(
                [
                    {
                        "Rank": i + 1,
                        "Retorno Anual (%)": p.annual_return * 100,
                        "Vol Anual (%)": p.annual_vol * 100,
                        "Sharpe": p.sharpe,
                        "Pesos": ", ".join(
                            f"{t}:{w*100:.1f}%" for t, w in p.weights.items()
                        ),
                    }
                    for i, p in enumerate(top_ports)
                ]
            ).set_index("Rank")

            st.subheader("🏆 Top 5 portfólios por Sharpe")
            st.dataframe(df_ports.style.format({
                "Retorno Anual (%)": "{:.2f}",
                "Vol Anual (%)": "{:.2f}",
                "Sharpe": "{:.2f}",
            }))

            df_all = pd.DataFrame(
                [
                    {
                        "Retorno Anual (%)": p.annual_return * 100,
                        "Vol Anual (%)": p.annual_vol * 100,
                        "Sharpe": p.sharpe,
                    }
                    for p in portfolios
                ]
            )

            st.subheader("📈 Distribuição de portfólios simulados (Retorno x Volatilidade)")
            st.scatter_chart(
                df_all,
                x="Vol Anual (%)",
                y="Retorno Anual (%)",
            )

    # ============= SIMULADOR DE INVESTIMENTO ============
    elif analysis_type == "Simulador de investimento (quanto rende?)":
        invested = st.number_input(
            "Valor investido (R$)",
            min_value=100.0,
            value=1000.0,
            step=100.0,
        )

        if st.button("💰 Calcular rendimento", key="btn_sim"):
            try:
                prices = download_prices(tickers, years=years)
            except Exception as e:
                st.error(f"Erro ao baixar preços: {e}")
                return

            try:
                simulation = simulate_investment(prices, invested)
            except Exception as e:
                st.error(f"Erro ao simular investimento: {e}")
                return

            st.subheader("📊 Ranking de rentabilidade (com base no período escolhido)")
            st.dataframe(
                simulation.style.format({
                    "Retorno (%)": "{:.2f}",
                    "Valor Final (R$)": "R${:,.2f}",
                })
            )

            st.subheader("📈 Evolução do valor investido ao longo do tempo")
            evolutions = prices / prices.iloc[0] * invested
            st.line_chart(evolutions)


if __name__ == "__main__":
    main()
