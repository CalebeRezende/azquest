from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ======================================================
#  Garantir yfinance instalado mesmo se o host ignorar
#  o requirements.txt (útil em alguns ambientes)
# ======================================================
try:
    import yfinance as yf
except ModuleNotFoundError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

TRADING_DAYS = 252

# ======================================================
#  UNIVERSO DE INVESTIMENTOS PRÉ-DEFINIDOS
#  (inclui B3 + EUA, default em EUA p/ evitar erro na nuvem)
# ======================================================

INVESTMENT_UNIVERSE: Dict[str, str] = {
    # B3
    "PETR4 (Petrobras PN)": "PETR4.SA",
    "VALE3 (Vale ON)": "VALE3.SA",
    "ITUB4 (Itaú Unibanco PN)": "ITUB4.SA",
    "BBDC4 (Bradesco PN)": "BBDC4.SA",
    "BBAS3 (Banco do Brasil ON)": "BBAS3.SA",
    "ABEV3 (Ambev ON)": "ABEV3.SA",
    "WEGE3 (Weg ON)": "WEGE3.SA",
    "MGLU3 (Magazine Luiza ON)": "MGLU3.SA",
    "BOVA11 (ETF Ibovespa)": "BOVA11.SA",
    "IVVB11 (ETF S&P 500 B3)": "IVVB11.SA",
    "^BVSP (Ibovespa índice)": "^BVSP",
    # EUA
    "AAPL (Apple)": "AAPL",
    "MSFT (Microsoft)": "MSFT",
    "AMZN (Amazon)": "AMZN",
    "GOOG (Alphabet)": "GOOG",
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


@dataclass
class PortfolioStats:
    weights: Dict[str, float]
    annual_return: float
    annual_vol: float
    sharpe: float


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
                # pega nível "Adj Close" no primeiro nível
                s = data["Adj Close"]
                if isinstance(s, pd.DataFrame):
                    # se tiver mais de uma coluna, tenta achar este ticker
                    if ticker in s.columns:
                        s = s[ticker]
                    else:
                        # pega a primeira coluna, pelo menos
                        s = s.iloc[:, 0]
            elif "Adj Close" in level1:
                # pega por nível=1
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
        # se por alguma razão sobrou DataFrame, pega a primeira col
        s = s.iloc[:, 0]

    s = s.dropna()
    if s.empty:
        return None

    return s


def download_prices(tickers: List[str], years: int = 3) -> pd.DataFrame:
    """
    Baixa preços 'Adj Close' via yfinance para 1 ou vários tickers.
    Baixa CADA TICKER separadamente para:
    - evitar erros globais quando um deles falha
    - conseguir ignorar ativos individuais com problema.
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
        except Exception as e:
            failed.append(t)
            continue

        s = _extract_adj_close_for_single_ticker(data, t)
        if s is None or s.empty:
            failed.append(t)
            continue

        series_map[t] = s

    if failed:
        st.warning(
            f"Não foi possível obter dados para: {', '.join(failed)}. "
            "Esses ativos serão ignorados na análise."
        )

    if not series_map:
        raise ValueError("Nenhum dado válido encontrado para os tickers informados.")

    # Alinha todas as séries pelo índice de datas
    df = pd.DataFrame(series_map)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("Nenhum dado válido após alinhamento e limpeza.")

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
            sharpe = (ann_return - risk_free) / ann_vol

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
    st.caption("Escolha uma lista de investimentos, período e valor, e veja qual teria sido melhor.")

    # --------- SIDEBAR: UNIVERSO DE INVESTIMENTOS ----------
    st.sidebar.header("🎯 Universo de investimentos")

    universe_labels = list(INVESTMENT_UNIVERSE.keys())

    # Default em EUA para evitar problema de B3 em alguns ambientes
    default_selection = [
        "AAPL (Apple)",
        "MSFT (Microsoft)",
        "SPY (ETF S&P 500)",
    ]

    selected_labels = st.sidebar.multiselect(
        "Selecione os ativos para analisar",
        options=universe_labels,
        default=[l for l in default_selection if l in universe_labels],
        help="Você pode escolher vários (ex: 5, 10, 15 ativos) para comparar.",
    )

    extra_tickers_str = st.sidebar.text_input(
        "Tickers adicionais (separados por vírgula, opcional)",
        value="",
        help="Ex: PETR4.SA, VALE3.SA, AAPL, MSFT, TSLA",
    )

    years = st.sidebar.slider(
        "Anos de histórico",
        min_value=1,
        max_value=10,
        value=3,
    )

    risk_free = st.sidebar.number_input(
        "Taxa livre de risco (ao ano, em %)",
        value=10.0,
        step=0.5,
    ) / 100.0

    selected_tickers = [INVESTMENT_UNIVERSE[label] for label in selected_labels]

    extra_tickers = [
        t.strip()
        for t in extra_tickers_str.split(",")
        if t.strip()
    ]

    tickers = selected_tickers + extra_tickers

    if len(tickers) == 0:
        st.warning("Selecione pelo menos 1 ativo na barra lateral ou informe tickers extras.")
        return

    # --------- MODO DE ANÁLISE ----------
    analysis_type = st.radio(
        "Tipo de análise",
        options=[
            "Estatísticas por ativo",
            "Simulação de portfólio (Monte Carlo)",
            "Simulador de investimento (quanto rende?)",
        ],
        horizontal=True,
    )

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

            st.subheader("📊 Ranking de rentabilidade (período escolhido)")
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
