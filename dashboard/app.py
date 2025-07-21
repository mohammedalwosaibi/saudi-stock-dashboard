import streamlit as st
import yfinance as yf
import numpy as np
import altair as alt

st.title("Saudi Stock Dashboard")
st.write("One of the biggest challenges novice investors face is not knowing whether their strategy is working. This application helps address this issue by comparing two of the most common strategies for some of the biggest Saudi stocks:")

st.markdown("""
* Buy-and-Hold - with dividend yields included
* Simple Moving Average (SMA) Crossover Strategy - use the moving averages to buy or sell
""")

st.markdown("An in-depth explanation and optimization of the SMA crossover strategy can be found in this [notebook](https://github.com/mohammedalwosaibi/saudi-stock-dashboard/blob/main/notebooks/aramco_strategy_analysis.ipynb).")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    company_name = st.selectbox("Company Name", [
        "Saudi Arabian Oil Company",
        "Al Rajhi Banking and Investment Corporation",
        "The Saudi National Bank",
        "Saudi Telecom Company",
        "Saudi Arabian Mining Company",
        "Alinma Bank",
        "Saudi Basic Industries Corporation",
        "Dr. Sulaiman Al Habib Medical Services Group Company",
        "Riyad Bank",
        "Saudi Awwal Bank",
        "Abdullah Al-Othaim Markets Company",
        ])

with col2:
    st.number_input("Short-Term SMA", 5, 50, 30, key="short_term_sma")

with col3:
    st.number_input("Long-Term SMA", 51, 140, 80, key="long_term_sma")

tickers = {
    "Saudi Arabian Oil Company": "2222.SR",
    "Al Rajhi Banking and Investment Corporation": "1120.SR",
    "The Saudi National Bank": "1180.SR",
    "Saudi Telecom Company": "7010.SR",
    "Saudi Arabian Mining Company": "1211.SR",
    "Alinma Bank": "1150.SR",
    "Saudi Basic Industries Corporation": "2010.SR",
    "Dr. Sulaiman Al Habib Medical Services Group Company": "4013.SR",
    "Riyad Bank": "1010.SR",
    "Saudi Awwal Bank": "1060.SR",
    "Abdullah Al-Othaim Markets Company": "4001.SR",
}

current_ticker = tickers[company_name]

current_stock = yf.download(current_ticker, "2019-12-01", "2025-07-13", multi_level_index=False, auto_adjust=False)
current_stock.reset_index(inplace=True)


dividend_yield = yf.Ticker(current_ticker).info.get("dividendYield")

if dividend_yield is None:
    dividend_yield = 0.00

STARTING_CASH = 100
DIVIDEND_MULTIPLIER = dividend_yield / 100 + 1
START_DATE = "2020-07-01"
END_DATE = "2025-07-01"

shares_owned = STARTING_CASH / current_stock["Open"][current_stock["Date"] == START_DATE].iloc[0]
final_shares_price = shares_owned * current_stock["Open"][current_stock["Date"] == END_DATE].iloc[0]
compound_annual_growth_multiplier = (final_shares_price / STARTING_CASH) ** (1/5)
compound_annual_growth_multiplier_with_dividends = compound_annual_growth_multiplier * DIVIDEND_MULTIPLIER

print(f"On {START_DATE}, {STARTING_CASH} SAR buys {shares_owned:.2f} shares of this stock.")
print(f"On {END_DATE}, these shares are worth {final_shares_price:.2f} SAR, showing a compound annual growth rate of {(compound_annual_growth_multiplier - 1) * 100:.2f}%.")
print(f"With dividends reinvested at a yield of {(DIVIDEND_MULTIPLIER - 1) * 100:.2f}%, the finalized annual return is {(compound_annual_growth_multiplier_with_dividends - 1) * 100:.2f}%.")

folds = [
    {"train_start": "2020-07-01", "train_end": "2021-07-01", "test_end": "2022-01-01"},
    {"train_start": "2021-01-01", "train_end": "2022-01-01", "test_end": "2022-07-01"},
    {"train_start": "2021-07-01", "train_end": "2022-07-01", "test_end": "2023-01-01"},
    {"train_start": "2022-01-01", "train_end": "2023-01-01", "test_end": "2023-07-01"},
    {"train_start": "2022-07-01", "train_end": "2023-07-01", "test_end": "2024-01-01"},
    {"train_start": "2023-01-01", "train_end": "2024-01-01", "test_end": "2024-07-01"},
    {"train_start": "2023-07-01", "train_end": "2024-07-01", "test_end": "2025-01-01"},
    {"train_start": "2024-01-01", "train_end": "2025-01-01", "test_end": "2025-07-01"},
]

final_yields = []

for fold_index, fold in enumerate(folds):
    
    train = current_stock[
        (current_stock["Date"] >= fold["train_start"]) &
        (current_stock["Date"] < fold["train_end"])
    ]
    
    test = current_stock[
        (current_stock["Date"] >= fold["train_end"]) &
        (current_stock["Date"] < fold["test_end"])
    ]

    sma_pair_yields = {}

    for short_term, long_term in ((10, 50), (20, 100), (50, 200), (21, 55), (9, 21)):

        short_term_sma = current_stock["Open"].rolling(short_term).mean()[
            (current_stock["Date"] >= fold["train_start"]) &
            (current_stock["Date"] < fold["train_end"])
        ]
        
        long_term_sma = current_stock["Open"].rolling(long_term).mean()[
            (current_stock["Date"] >= fold["train_start"]) &
            (current_stock["Date"] < fold["train_end"])
        ]

        current_cash = STARTING_CASH
        current_shares = 0

        if short_term_sma.iloc[0] > long_term_sma.iloc[0]:
            current_shares = current_cash / train.iloc[0]["Open"]
            current_cash = 0

        for i in range(1, len(train)):
            if short_term_sma.iloc[i] > long_term_sma.iloc[i] and short_term_sma.iloc[i - 1] < long_term_sma.iloc[i - 1]:
                current_shares = current_cash / train.iloc[i]["Open"]
                current_cash = 0
            if short_term_sma.iloc[i] < long_term_sma.iloc[i] and short_term_sma.iloc[i - 1] > long_term_sma.iloc[i - 1]:
                current_cash = current_shares * train.iloc[i]["Open"]
                current_shares = 0

        sma_pair_yields[current_cash + current_shares * train.iloc[-1]["Open"]] = (short_term, long_term)

    short_term, long_term = sma_pair_yields[np.max(list(sma_pair_yields.keys()))]
    
    print(f"The best performing SMA parameter pair during Fold #{fold_index} was ({short_term}, {long_term}) at a cumulative (and annualized) yield of {np.max(list(sma_pair_yields.keys())) - 100:.2f}%.")

    short_term_sma = current_stock["Open"].rolling(short_term).mean()[
        (current_stock["Date"] >= fold["train_end"]) &
        (current_stock["Date"] < fold["test_end"])
    ]
    
    long_term_sma = current_stock["Open"].rolling(long_term).mean()[
        (current_stock["Date"] >= fold["train_end"]) &
        (current_stock["Date"] < fold["test_end"])
    ]

    current_cash = 100
    current_shares = 0

    if short_term_sma.iloc[0] > long_term_sma.iloc[0]:
        current_shares = current_cash / test.iloc[0]["Open"]
        current_cash = 0

    for i in range(1, len(test)):
        if short_term_sma.iloc[i] > long_term_sma.iloc[i] and short_term_sma.iloc[i - 1] < long_term_sma.iloc[i - 1]:
            current_shares = current_cash / test.iloc[i]["Open"]
            current_cash = 0
        if short_term_sma.iloc[i] < long_term_sma.iloc[i] and short_term_sma.iloc[i - 1] > long_term_sma.iloc[i - 1]:
            current_cash = current_shares * test.iloc[i]["Open"]
            current_shares = 0

    final_yield = max(current_cash, current_shares * test.iloc[-1]["Open"]) / 100
    print((current_cash + current_shares * test.iloc[-1]["Open"]) / 100)
    
    print(f"On the test set, this pair yielded a cumulative yield of {(final_yield - 1) * 100:.2f}% over a 6-month period.\n")

    final_yields.append(final_yield)

print(f"Final Cumulative Yield: {(np.prod(final_yields) - 1) * 100:.2f}%.")
print(f"Final Annualized Yield: {(np.prod(final_yields) ** (1/4) - 1) * 100:.2f}%.")

current_stock["Short-Term SMA"] = current_stock["Open"].rolling(st.session_state.short_term_sma).mean()
current_stock["Long-Term SMA"] = current_stock["Open"].rolling(st.session_state.long_term_sma).mean()

plot_df = current_stock[
    (current_stock["Date"] >= START_DATE) & 
    (current_stock["Date"] <= END_DATE)
][["Date", "Open", "Short-Term SMA", "Long-Term SMA"]]

plot_df = plot_df.melt(
    id_vars="Date",
    var_name="Type",
    value_name="Price"
)

y_min = plot_df["Price"].min()
y_max = plot_df["Price"].max()

chart = (
    alt.Chart(plot_df)
    .mark_line(strokeWidth=2)
    .encode(
        x="Date:T",
        y=alt.Y("Price:Q", scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color(
            "Type:N",
            legend=alt.Legend(title="Series"),
            scale=alt.Scale(range=["#FFD700", "#0044AA", "#DD7722"])
        )
    )
    .configure_legend(
        labelColor='white',
        titleColor='white',
        symbolType='stroke'
    )
    .properties(
        title="Stock Price + Short-Term & Long-Term SMA"
    )
)

st.altair_chart(chart, use_container_width=True)

current_cash = STARTING_CASH
current_shares = 0

current_stock_date_adjusted = current_stock[
    (current_stock["Date"] >= START_DATE) & (current_stock["Date"] <= END_DATE)
]

short_term_sma = current_stock["Open"].rolling(st.session_state.short_term_sma).mean()[
    (current_stock["Date"] >= START_DATE) & (current_stock["Date"] <= END_DATE)
]
long_term_sma = current_stock["Open"].rolling(st.session_state.long_term_sma).mean()[
    (current_stock["Date"] >= START_DATE) & (current_stock["Date"] <= END_DATE)
]

if short_term_sma.iloc[0] > long_term_sma.iloc[0]:
    current_shares = current_cash / current_stock_date_adjusted["Open"].iloc[0]
    current_cash = 0

for i in range(1, len(short_term_sma)):
    if short_term_sma.iloc[i] > long_term_sma.iloc[i] and short_term_sma.iloc[i - 1] < long_term_sma.iloc[i - 1] and current_cash != 0:
        current_shares = current_cash / current_stock_date_adjusted["Open"].iloc[i]
        current_cash = 0
    elif short_term_sma.iloc[i] < long_term_sma.iloc[i] and short_term_sma.iloc[i - 1] > long_term_sma.iloc[i - 1] and current_cash == 0:
        current_cash = current_shares * current_stock_date_adjusted["Open"].iloc[i]
        current_shares = 0

final_yield = max(current_cash, current_shares * current_stock_date_adjusted["Open"].iloc[-1]) / 100

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**<p style='padding-top: 0; text-align: center; color: #ddd;'>Selected SMA Parameter Pair Annualized Return</p>**", unsafe_allow_html=True)
    st.markdown(f"<h3 style='padding-top: 0; margin-left: 1rem; text-align: center;'>{final_yield:.2f}%</h2>", unsafe_allow_html=True)

with info_col2:
    st.markdown("**<p style='padding-top: 0; text-align: center; color: #ddd;'>Buy-and-Hold Annualized Return</p>**", unsafe_allow_html=True)
    st.markdown(f"<h3 style='padding-top: 0; margin-left: 1rem; text-align: center;'>{(compound_annual_growth_multiplier_with_dividends - 1) * 100:.2f}%</h2>", unsafe_allow_html=True)

with info_col3:
    st.markdown("**<p style='padding-top: 0; text-align: center; color: #ddd;'>Validated SMA Crossover Annualized Return</p>**", unsafe_allow_html=True)
    st.markdown(f"<h3 style='padding-top: 0; margin-left: 1rem; text-align: center;'>{(np.prod(final_yields) ** (1/4) - 1) * 100:.2f}%</h2>", unsafe_allow_html=True)

st.markdown("<h3 style='padding-top: 2rem; text-align: center;'>Additional Information</h3>", unsafe_allow_html=True)

st.write("The Buy-and-Hold strategy dominates in the Saudi Arabian stock market due to the substantial dividend yields and the low trend-persistence resulting in many false signals from the SMA crossover strategy. For more information about how the validated SMA annualized return is calculated, have a quick glance at this [notebook](https://github.com/mohammedalwosaibi/saudi-stock-dashboard/blob/main/notebooks/aramco_strategy_analysis.ipynb).")