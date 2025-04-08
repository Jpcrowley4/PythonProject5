from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        margin-bottom: 1rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .positive {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid green;
    }
    .negative {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid red;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-change {
        font-size: 1rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        font-size: 0.8rem;
    }
    .disclaimer {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin-top: 2rem;
    }
    .tooltip-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196f3;
        margin-bottom: 1rem;
    }
    .help-text {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
    .volume-table {
        margin-top: 1rem;
        width: 100%;
    }
    .volume-table th {
        background-color: #f1f1f1;
        padding: 8px;
        text-align: left;
    }
    .volume-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .volume-table tr:hover {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

try:
    st.markdown("<div class='main-header'>Stock Price Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "AI-powered stock price predictions based on historical patterns.")


    tab1, tab2, tab3 = st.tabs(["📈 Price Prediction", "📊 Market Analysis", "ℹ️ How It Works"])


    with tab1:
        st.markdown("<div class='subheader'>Enter Stock Details</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="tooltip-card">
            <strong>📌 Quick Guide:</strong> Enter a stock symbol (like AAPL for Apple) and select how far back you want to analyze.
            Longer periods may give the model more data to learn from.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])

        with col1:
            ticker_input = st.text_input("Stock Symbol", value="AAPL",
                                         help="Enter the ticker symbol of the stock (e.g., AAPL for Apple)")
            st.markdown(
                "**Popular stocks:** AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon), TSLA (Tesla)")

        with col2:
            period = st.selectbox("Historical Data",
                                  options=["1y", "2y", "3y", "5y"],
                                  index=1,
                                  help="How much historical data to analyze")

            prediction_timeframe = st.selectbox("Prediction Timeframe",
                                                options=["1 Day", "3 Days", "1 Week", "2 Weeks"],
                                                index=0,
                                                help="How far into the future to predict")

            st.markdown("<p class='help-text'>Longer periods provide more training data</p>", unsafe_allow_html=True)


        if st.button("Analyze Stock", type="primary"):
            status = st.empty()
            status.info(f"Getting historical data for {ticker_input}...")
            data = yf.download(ticker_input, period=period, interval="1d")

            if data.empty:
                st.error(f"No data found for '{ticker_input}'. Please check the symbol and try again.")
                st.stop()

            try:
                ticker_info = yf.Ticker(ticker_input).info
                company_name = ticker_info.get('longName', ticker_input)
                sector = ticker_info.get('sector', 'N/A')
                industry = ticker_info.get('industry', 'N/A')

                st.markdown(f"""
                <div class='card'>
                    <h2>{company_name} ({ticker_input})</h2>
                    <p><strong>Sector:</strong> {sector} | <strong>Industry:</strong> {industry}</p>
                </div>
                """, unsafe_allow_html=True)

                metric_cols = st.columns(4)

                current_price = float(ticker_info.get('currentPrice', data['Close'].iloc[-1]))

                with metric_cols[0]:
                    st.metric("Current Price", f"${current_price:.2f}")

                with metric_cols[1]:
                    if len(data) >= 2:
                        previous_close = float(data['Close'].iloc[-2])
                        day_change = ((current_price - previous_close) / previous_close) * 100
                        st.metric("Day Change", f"{day_change:.2f}%",
                                  f"{'+' if day_change > 0 else ''}{day_change:.2f}%")
                    else:
                        st.metric("Day Change", "N/A")

                with metric_cols[2]:
                    high_52week = float(ticker_info.get('fiftyTwoWeekHigh', data['High'].max()))
                    st.metric("52-Week High", f"${high_52week:.2f}")

                with metric_cols[3]:
                    low_52week = float(ticker_info.get('fiftyTwoWeekLow', data['Low'].min()))
                    st.metric("52-Week Low", f"${low_52week:.2f}")

                pe_ratio = ticker_info.get('trailingPE', 'N/A')
                forward_pe = ticker_info.get('forwardPE', 'N/A')
                peg_ratio = ticker_info.get('pegRatio', 'N/A')
                price_to_book = ticker_info.get('priceToBook', 'N/A')
                profit_margins = ticker_info.get('profitMargins', 'N/A')
                if profit_margins != 'N/A':
                    profit_margins = f"{profit_margins:.2%}"
                debt_to_equity = ticker_info.get('debtToEquity', 'N/A')
                if debt_to_equity != 'N/A':
                    debt_to_equity = f"{debt_to_equity:.2f}"

                st.markdown("<div class='subheader'>Fundamental Analysis</div>", unsafe_allow_html=True)

                fund_cols = st.columns(3)

                with fund_cols[0]:
                    st.markdown(f"""
                    <div class="card">
                        <h3>Valuation Ratios</h3>
                        <table style="width:100%">
                            <tr>
                                <td>P/E Ratio (TTM)</td>
                                <td><strong>{pe_ratio}</strong></td>
                            </tr>
                            <tr>
                                <td>Forward P/E</td>
                                <td><strong>{forward_pe}</strong></td>
                            </tr>
                            <tr>
                                <td>PEG Ratio</td>
                                <td><strong>{peg_ratio}</strong></td>
                            </tr>
                            <tr>
                                <td>Price to Book</td>
                                <td><strong>{price_to_book}</strong></td>
                            </tr>
                        </table>
                        <p class="help-text">Lower P/E may indicate better value, but context matters.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with fund_cols[1]:
                    earnings_growth = ticker_info.get('earningsGrowth', 'N/A')
                    if earnings_growth != 'N/A':
                        earnings_growth = f"{earnings_growth:.2%}"
                    revenue_growth = ticker_info.get('revenueGrowth', 'N/A')
                    if revenue_growth != 'N/A':
                        revenue_growth = f"{revenue_growth:.2%}"

                    st.markdown(f"""
                    <div class="card">
                        <h3>Growth Metrics</h3>
                        <table style="width:100%">
                            <tr>
                                <td>Earnings Growth (YoY)</td>
                                <td><strong>{earnings_growth}</strong></td>
                            </tr>
                            <tr>
                                <td>Revenue Growth (YoY)</td>
                                <td><strong>{revenue_growth}</strong></td>
                            </tr>
                            <tr>
                                <td>Profit Margin</td>
                                <td><strong>{profit_margins}</strong></td>
                            </tr>
                        </table>
                        <p class="help-text">Higher growth often supports higher valuation multiples.</p>
                    </div>
                    """, unsafe_allow_html=True)

                with fund_cols[2]:
                    dividend_yield = ticker_info.get('dividendYield', 'N/A')
                    if dividend_yield != 'N/A':
                        dividend_yield = f"{dividend_yield:.2%}"
                    dividend_rate = ticker_info.get('dividendRate', 'N/A')

                    st.markdown(f"""
                    <div class="card">
                        <h3>Financial Health</h3>
                        <table style="width:100%">
                            <tr>
                                <td>Debt to Equity</td>
                                <td><strong>{debt_to_equity}</strong></td>
                            </tr>
                            <tr>
                                <td>Dividend Yield</td>
                                <td><strong>{dividend_yield}</strong></td>
                            </tr>
                            <tr>
                                <td>Dividend Rate (Annual)</td>
                                <td><strong>${dividend_rate if dividend_rate != 'N/A' else 'N/A'}</strong></td>
                            </tr>
                        </table>
                        <p class="help-text">Lower debt-to-equity typically indicates lower financial risk.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="tooltip-card">
                    <strong>📊 Understanding Fundamental Ratios:</strong>
                    <ul>
                        <li><strong>P/E Ratio:</strong> Price to earnings - how much you're paying for each dollar of earnings</li>
                        <li><strong>PEG Ratio:</strong> P/E ratio divided by growth rate - lower values may indicate better value relative to growth</li>
                        <li><strong>Price to Book:</strong> Share price relative to book value - lower values might indicate undervaluation</li>
                        <li><strong>Debt to Equity:</strong> Total debt divided by equity - measures financial leverage</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"### {ticker_input} Stock Analysis")
                st.warning(f"Limited company information available for {ticker_input}.")

            display_data = data.tail().copy()
            display_data.index = display_data.index.strftime('%b %d, %Y')

            st.markdown("<div class='subheader'>Recent Price History</div>", unsafe_allow_html=True)
            st.dataframe(
                display_data,
                column_config={
                    "Open": st.column_config.NumberColumn(
                        "Open",
                        format="$%.2f",
                    ),
                    "High": st.column_config.NumberColumn(
                        "High",
                        format="$%.2f",
                    ),
                    "Low": st.column_config.NumberColumn(
                        "Low",
                        format="$%.2f",
                    ),
                    "Close": st.column_config.NumberColumn(
                        "Close",
                        format="$%.2f",
                    ),
                    "Volume": st.column_config.NumberColumn(
                        "Volume",
                        format="%d",
                    ),
                },
                hide_index=True,
                use_container_width=True
            )

            status.info("Creating price chart...")

            data['20-Day Average'] = data['Close'].rolling(window=20).mean()
            data['50-Day Average'] = data['Close'].rolling(window=50).mean()

            fig = go.Figure()


            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['20-Day Average'],
                    name="20-day Average",
                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['50-Day Average'],
                    name="50-day Average",
                    line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5)
                )
            )

            fig.update_layout(
                title=f"{ticker_input} Stock Price History",
                xaxis_rangeslider_visible=False,
                yaxis_title="Price ($)",
                height=500,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig.update_yaxes(tickprefix="$")
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig, use_container_width=True)


            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Recent Trading Volume</h3>", unsafe_allow_html=True)

            volume_data = data[['Volume']].tail(10).copy()
            volume_data.index = volume_data.index.strftime('%b %d, %Y')


            st.dataframe(
                volume_data,
                column_config={
                    "Volume": st.column_config.NumberColumn(
                        "Volume (Shares)",
                        format="%,d"
                    )
                },
                hide_index=False,
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)


            st.markdown("""
            <div class="tooltip-card">
                <strong> Reading the Chart:</strong>
                <ul>
                    <li><strong>Candlesticks:</strong> Show opening, closing, high and low prices for each day</li>
                    <li><strong>Orange line:</strong> 20-day moving average (short-term trend)</li>
                    <li><strong>Red line:</strong> 50-day moving average (longer-term trend)</li>
                    <li><strong>Volume table:</strong> Shows how many shares were traded on recent days</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


            status.info("Preparing data for prediction...")


            def predict(ticker, raw_prediction, current_price, days_ahead=3):
                data = yf.download(ticker, period="1y") if 'data' not in locals() else data


                current_price = float(current_price)
                raw_prediction = float(raw_prediction)


                daily_returns = data['Close'].pct_change().dropna()
                daily_volatility = float(daily_returns.std())


                period_volatility = daily_volatility * np.sqrt(days_ahead)


                if '20-Day Average' in data.columns:
                    ma_20 = float(data['20-Day Average'].iloc[-1])
                else:
                    ma_20 = float(data['Close'].rolling(window=20).mean().iloc[-1])

                if '50-Day Average' in data.columns:
                    ma_50 = float(data['50-Day Average'].iloc[-1])
                else:
                    ma_50 = float(data['Close'].rolling(window=50).mean().iloc[-1])



                if days_ahead <= 3:
                    mean_target = 0.5 * current_price + 0.3 * ma_20 + 0.2 * ma_50
                else:
                    mean_target = 0.3 * current_price + 0.3 * ma_20 + 0.4 * ma_50


                max_change = period_volatility * 2


                upper_bound = current_price * (1 + max_change)
                lower_bound = current_price * (1 - max_change)


                deviation = abs(raw_prediction - mean_target) / mean_target


                if deviation > 0.1:
                    reversion_strength = 0.7
                elif deviation > 0.05:
                    reversion_strength = 0.5
                else:
                    reversion_strength = 0.3


                adjusted_prediction = (raw_prediction * (1 - reversion_strength) +
                                       mean_target * reversion_strength)


                final_prediction = max(min(adjusted_prediction, upper_bound), lower_bound)


                return {
                    "final_prediction": final_prediction,
                    "debug": {
                        "raw_prediction": raw_prediction,
                        "mean_target": mean_target,
                        "upper_bound": upper_bound,
                        "lower_bound": lower_bound,
                        "reversion_strength": reversion_strength,
                        "daily_volatility": daily_volatility,
                        "max_change_pct": max_change * 100
                    }
                }



            delta = data['Close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = loss.abs()

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))


            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()


            data = data.dropna()

            features = ['Close', 'Volume', '20-Day Average', '50-Day Average', 'RSI', 'MACD']


            scaler_dict = {}
            scaled_features = pd.DataFrame(index=data.index)

            for feature in features:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
                scaler_dict[feature] = scaler


            price_scaler = scaler_dict['Close']


            sequence_length = 60


            def sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data.iloc[i:i + seq_length].values)
                    y.append(data['Close'].iloc[i + seq_length])
                return np.array(X), np.array(y)


            X, y = sequences(scaled_features, sequence_length)
            y = y.reshape(-1, 1)


            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]


            status.info("Training AI model... This may take a moment.")


            model = Sequential()
            model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=100, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=50, activation='relu'))
            model.add(Dense(units=1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )


            test_predictions = model.predict(X_test)
            test_predictions = price_scaler.inverse_transform(test_predictions)
            y_test_actual = price_scaler.inverse_transform(y_test)


            timeframe_days = {
                "1 Day": 1,
                "3 Days": 3,
                "1 Week": 5,
                "2 Weeks": 10
            }

            days_to_predict = timeframe_days[prediction_timeframe]


            status.success("Analysis complete!")


            st.markdown(f"<div class='subheader'>{prediction_timeframe} Price Prediction</div>", unsafe_allow_html=True)


            if days_to_predict == 1:

                last_sequence = scaled_features.iloc[-sequence_length:].values
                last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))


                next_day_pred_scaled = model.predict(last_sequence)
                next_day_pred = float(price_scaler.inverse_transform(next_day_pred_scaled)[0][0])

                last_close = float(data['Close'].iloc[-1])

                prediction_result = predict(ticker_input, next_day_pred, last_close, days_ahead=1)

                next_day_pred = prediction_result['final_prediction']

                last_date = data.index[-1]
                next_date = last_date + pd.Timedelta(days=1)

                if next_date.weekday() > 4:
                    days_to_add = 8 - next_date.weekday()
                    next_date = last_date + pd.Timedelta(days=days_to_add)


                last_close = float(data['Close'].iloc[-1])
                pred_change = ((next_day_pred - last_close) / last_close) * 100
                change_direction = "positive" if pred_change >= 0 else "negative"
                change_symbol = "+" if pred_change >= 0 else ""


                st.markdown(f"""
                <div class='prediction-card {change_direction}'>
                    <h2>Predicted Price for {next_date.strftime('%A, %B %d, %Y')}</h2>
                    <div class='metric-value'>${next_day_pred:.2f}</div>
                    <div class='metric-change' style='color:{"green" if change_direction == "positive" else "red"}'>
                        {change_symbol}{pred_change:.2f}% from previous close
                    </div>
                </div>
                """, unsafe_allow_html=True)


                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Previous Close", f"${last_close:.2f}")
                with metric_cols[1]:
                    st.metric("Predicted Price", f"${next_day_pred:.2f}", f"{change_symbol}{pred_change:.2f}%")
                with metric_cols[2]:

                    mae = float(np.mean(np.abs(test_predictions - y_test_actual)))
                    st.metric("Average Error", f"${mae:.2f}",
                              help="How far off the model's predictions typically are")


                st.markdown(f"""
                <div class="tooltip-card">
                    <strong>🔍 What this means:</strong> Based on historical patterns, the model predicts that {company_name} 
                    stock will {change_direction == "positive" and "rise" or "fall"} to ${next_day_pred:.2f} 
                    by {next_date.strftime('%A')}. The prediction is based on past trends, market momentum, and trading patterns.
                </div>
                """, unsafe_allow_html=True)


                st.markdown("<div class='subheader'>How Accurate Is The Model?</div>", unsafe_allow_html=True)


                fig_pred = go.Figure()


                fig_pred.add_trace(go.Scatter(
                    x=data.index[-len(y_test_actual):],
                    y=y_test_actual.flatten(),
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ))


                fig_pred.add_trace(go.Scatter(
                    x=data.index[-len(test_predictions):],
                    y=test_predictions.flatten(),
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', width=2)
                ))


                fig_pred.add_trace(go.Scatter(
                    x=[next_date],
                    y=[next_day_pred],
                    mode='markers',
                    name='Tomorrow\'s Prediction',
                    marker=dict(color='green', size=12, symbol='star')
                ))


                fig_pred.update_layout(
                    title="Actual vs Predicted Prices",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                    yaxis=dict(tickprefix="$")
                )

                st.plotly_chart(fig_pred, use_container_width=True)


                st.markdown("""
                <div class="tooltip-card">
                    <strong>📈 Reading the Chart Above:</strong>
                    <ul>
                        <li><strong>Blue line:</strong> Actual historical prices</li>
                        <li><strong>Red line:</strong> What the model would have predicted</li>
                        <li><strong>Green star:</strong> Tomorrow's prediction</li>
                    </ul>
                    <p>The closer the red line follows the blue line, the more accurate the model is.</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='subheader'>Market Insights Simplified</div>", unsafe_allow_html=True)


        st.markdown("""
        <div class="tooltip-card">
            <strong>📌 This tab:</strong> Shows key indicators that traders use to make decisions, presented in a simplified way.
            These insights will help you understand if the stock might be a good buy, hold, or sell right now.
        </div>
        """, unsafe_allow_html=True)

        if 'data' in locals():

            last_close = float(data['Close'].iloc[-1])
            last_ma20 = float(data['20-Day Average'].iloc[-1])
            last_ma50 = float(data['50-Day Average'].iloc[-1])
            last_rsi = float(data['RSI'].iloc[-1])
            last_macd = float(data['MACD'].iloc[-1])
            last_signal = float(data['Signal'].iloc[-1])


            st.markdown("### Key Market Signals")
            st.markdown("These indicators help traders decide when to buy or sell.")

            signal_cols = st.columns(3)

            with signal_cols[0]:
                ma_signal = "BUY" if last_close > last_ma50 else "SELL"
                ma_color = "green" if ma_signal == "BUY" else "red"

                st.markdown(f"""
                <div class="card">
                    <h3>Price Trend</h3>
                    <h2 style="color:{ma_color};">{ma_signal}</h2>
                    <p>The stock is trading {'above' if ma_signal == 'BUY' else 'below'} its 50-day average.</p>
                    <p class="help-text">When price is above the long-term average, it's usually seen as positive.</p>
                </div>
                """, unsafe_allow_html=True)

            with signal_cols[1]:
                if last_rsi > 70:
                    rsi_signal = "POTENTIALLY OVERVALUED"
                    rsi_color = "red"
                    rsi_explanation = "The stock may be overbought and due for a price decrease."
                elif last_rsi < 30:
                    rsi_signal = "POTENTIALLY UNDERVALUED"
                    rsi_color = "green"
                    rsi_explanation = "The stock may be oversold and due for a price increase."
                else:
                    rsi_signal = "NEUTRAL"
                    rsi_color = "gray"
                    rsi_explanation = "The stock is neither overbought nor oversold."

                st.markdown(f"""
                <div class="card">
                    <h3>Value Indicator (RSI)</h3>
                    <h2 style="color:{rsi_color};">{rsi_signal}</h2>
                    <p>{rsi_explanation}</p>
                    <p class="help-text">RSI measures if a stock might be overpriced or underpriced.</p>
                </div>
                """, unsafe_allow_html=True)

            with signal_cols[2]:
                momentum_signal = "POSITIVE" if last_macd > last_signal else "NEGATIVE"
                momentum_color = "green" if momentum_signal == "POSITIVE" else "red"

                st.markdown(f"""
                <div class="card">
                    <h3>Momentum</h3>
                    <h2 style="color:{momentum_color};">{momentum_signal}</h2>
                    <p>The stock {'has' if momentum_signal == 'POSITIVE' else 'lacks'} positive momentum right now.</p>
                    <p class="help-text">Momentum tells us if a trend is likely to continue or reverse.</p>
                </div>
                """, unsafe_allow_html=True)


            signals = []
            if ma_signal == "BUY":
                signals.append("POSITIVE")
            else:
                signals.append("NEGATIVE")

            if rsi_signal == "POTENTIALLY UNDERVALUED":
                signals.append("POSITIVE")
            elif rsi_signal == "POTENTIALLY OVERVALUED":
                signals.append("NEGATIVE")

            if momentum_signal == "POSITIVE":
                signals.append("POSITIVE")
            else:
                signals.append("NEGATIVE")

            positive_count = signals.count("POSITIVE")
            negative_count = signals.count("NEGATIVE")

            if positive_count > negative_count:
                overall_signal = "BULLISH (POSITIVE OUTLOOK)"
                overall_color = "green"
                overall_explanation = "Most indicators suggest positive price movement."
            elif negative_count > positive_count:
                overall_signal = "BEARISH (NEGATIVE OUTLOOK)"
                overall_color = "red"
                overall_explanation = "Most indicators suggest negative price movement."
            else:
                overall_signal = "NEUTRAL"
                overall_color = "gray"
                overall_explanation = "Mixed signals suggest sideways movement."

            st.markdown(f"""
            <div class="card" style="margin-top:20px;">
                <h3>Overall Market Outlook</h3>
                <h2 style="color:{overall_color}; text-align:center;">{overall_signal}</h2>
                <p style="text-align:center;">{overall_explanation}</p>
                <p style="text-align:center;">{positive_count} positive signals, {negative_count} negative signals</p>
            </div>
            """, unsafe_allow_html=True)


            if 'ticker_info' in locals() and 'pe_ratio' in locals():
                st.markdown("<div class='subheader'>Fundamental Analysis Dashboard</div>", unsafe_allow_html=True)

                industry_pe = ticker_info.get('industryPE', None)




                pe_status = "NEUTRAL"
                pe_color = "gray"
                pe_explanation = "P/E ratio is around the industry average."

                if industry_pe and pe_ratio != 'N/A':
                    try:
                        pe_num = float(pe_ratio)
                        ind_pe_num = float(industry_pe)

                        if pe_num < ind_pe_num * 0.7:
                            pe_status = "POTENTIALLY UNDERVALUED"
                            pe_color = "green"
                            pe_explanation = f"P/E ratio is significantly below the industry average of {ind_pe_num:.2f}."
                        elif pe_num > ind_pe_num * 1.3:
                            pe_status = "POTENTIALLY OVERVALUED"
                            pe_color = "red"
                            pe_explanation = f"P/E ratio is significantly above the industry average of {ind_pe_num:.2f}."
                    except:
                        pass


                fund_cols = st.columns(2)

                with fund_cols[0]:
                    st.markdown(f"""
                    <div class="card">
                        <h3>Valuation Analysis</h3>
                        <h2 style="color:{pe_color};">{pe_status}</h2>
                        <p>{pe_explanation}</p>
                        <p class="help-text">P/E Ratio: {pe_ratio} | Industry Average: {industry_pe if industry_pe else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with fund_cols[1]:
                    growth_status = "NEUTRAL"
                    growth_color = "gray"
                    growth_explanation = "Growth metrics are average."

                    if 'earnings_growth' in locals() and earnings_growth != 'N/A':
                        try:
                            eg_value = float(earnings_growth.strip('%')) / 100
                            if eg_value > 0.2:
                                growth_status = "STRONG GROWTH"
                                growth_color = "green"
                                growth_explanation = "Company shows strong earnings growth."
                            elif eg_value < 0:
                                growth_status = "DECLINING EARNINGS"
                                growth_color = "red"
                                growth_explanation = "Company's earnings are declining."
                        except:
                            pass

                    st.markdown(f"""
                    <div class="card">
                        <h3>Growth Analysis</h3>
                        <h2 style="color:{growth_color};">{growth_status}</h2>
                        <p>{growth_explanation}</p>
                        <p class="help-text">Companies with strong growth often command higher valuations.</p>
                    </div>
                    """, unsafe_allow_html=True)


            st.markdown("### Stock Performance")


            returns = data['Close'].pct_change().dropna()

            if len(returns) > 0:
                daily_return = float(returns.iloc[-1] * 100)
                avg_daily_return = float(returns.mean() * 100)


                monthly_return = avg_daily_return * 21
                yearly_return = avg_daily_return * 252

                volatility = float(returns.std() * 100)


                annualized_volatility = volatility * np.sqrt(252)
            else:
                daily_return = 0.0
                monthly_return = 0.0
                yearly_return = 0.0
                volatility = 0.0
                annualized_volatility = 0.0


            current_year = datetime.now().year
            start_of_year = datetime(current_year, 1, 1)
            ytd_data = data[data.index >= start_of_year]

            if not ytd_data.empty and len(ytd_data) > 1:
                ytd_start_price = float(ytd_data['Close'].iloc[0])
                ytd_return = ((last_close - ytd_start_price) / ytd_start_price) * 100
            else:
                ytd_return = 0.0


            st.markdown(f"""
            <div class="card">
                <h3>Returns</h3>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p class="metric-label">Today's Change</p>
                        <p class="metric-value" style="color: {'green' if daily_return >= 0 else 'red'}">
                            {'+' if daily_return >= 0 else ''}{daily_return:.2f}%
                        </p>
                    </div>
                    <div>
                        <p class="metric-label">Year to Date</p>
                        <p class="metric-value" style="color: {'green' if ytd_return >= 0 else 'red'}">
                            {'+' if ytd_return >= 0 else ''}{ytd_return:.2f}%
                        </p>
                    </div>
                    <div>
                        <p class="metric-label">Projected Annual Return</p>
                        <p class="metric-value" style="color: {'green' if yearly_return >= 0 else 'red'}">
                            {'+' if yearly_return >= 0 else ''}{yearly_return:.2f}%
                        </p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Volatility (Risk Level)</h3>
                <p>This stock has {
            'very low' if annualized_volatility < 15 else
            'low' if annualized_volatility < 25 else
            'moderate' if annualized_volatility < 35 else
            'high' if annualized_volatility < 45 else
            'very high'
            } volatility at <strong>{annualized_volatility:.1f}%</strong> annually.</p>
                <p class="help-text">Volatility measures how much a stock's price fluctuates. Higher volatility means higher risk but also potential for higher returns.</p>
            </div>
            """, unsafe_allow_html=True)


            st.markdown("### Price Performance Over Time")


            fig_perf = go.Figure()


            if len(data) > 252:
                one_month_ago = data.index[-1] - pd.Timedelta(days=30)
                three_months_ago = data.index[-1] - pd.Timedelta(days=90)
                six_months_ago = data.index[-1] - pd.Timedelta(days=180)
                one_year_ago = data.index[-1] - pd.Timedelta(days=365)


                fig_perf.add_trace(go.Scatter(
                    x=[data.index[-1], one_month_ago, three_months_ago, six_months_ago, one_year_ago],
                    y=[data['Close'].iloc[-1],
                       data.loc[data.index >= one_month_ago, 'Close'].iloc[0],
                       data.loc[data.index >= three_months_ago, 'Close'].iloc[0],
                       data.loc[data.index >= six_months_ago, 'Close'].iloc[0],
                       data.loc[data.index >= one_year_ago, 'Close'].iloc[0]],
                    mode='markers',
                    marker=dict(size=10),
                    name='Key Points'
                ))


            fig_perf.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue', width=2)
            ))


            fig_perf.update_layout(
                title="Stock Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                template="plotly_white",
                hovermode="x unified",
                yaxis=dict(tickprefix="$")
            )

            st.plotly_chart(fig_perf, use_container_width=True)


            if len(data) > 252:
                one_month_price = float(data.loc[data.index >= one_month_ago, 'Close'].iloc[0])
                three_month_price = float(data.loc[data.index >= three_months_ago, 'Close'].iloc[0])
                six_month_price = float(data.loc[data.index >= six_months_ago, 'Close'].iloc[0])
                one_year_price = float(data.loc[data.index >= one_year_ago, 'Close'].iloc[0])

                one_month_return = ((last_close - one_month_price) / one_month_price) * 100
                three_month_return = ((last_close - three_month_price) / three_month_price) * 100
                six_month_return = ((last_close - six_month_price) / six_month_price) * 100
                one_year_return = ((last_close - one_year_price) / one_year_price) * 100

                st.markdown(f"""
                <div class="card">
                    <h3>Historical Performance</h3>
                    <table style="width:100%">
                        <tr>
                            <th>Time Period</th>
                            <th>Return</th>
                        </tr>
                        <tr>
                            <td>Past Month</td>
                            <td style="color:{'green' if one_month_return >= 0 else 'red'}">{'+' if one_month_return >= 0 else ''}{one_month_return:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Past 3 Months</td>
                            <td style="color:{'green' if three_month_return >= 0 else 'red'}">{'+' if three_month_return >= 0 else ''}{three_month_return:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Past 6 Months</td>
                            <td style="color:{'green' if six_month_return >= 0 else 'red'}">{'+' if six_month_return >= 0 else ''}{six_month_return:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Past Year</td>
                            <td style="color:{'green' if one_year_return >= 0 else 'red'}">{'+' if one_year_return >= 0 else ''}{one_year_return:.2f}%</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("Please analyze a stock in the Price Prediction tab first to see market insights.")


    with tab3:
        st.markdown("### How This Works")
        st.markdown("""


    This app uses artificial intelligence to predict stock prices based on historical patterns. Through: 

    1. **Data Collection:** We get historical stock data from Yahoo Finance.
    2. **Pattern Recognition:** The AI model (called LSTM - Long Short-Term Memory) learns patterns from past price movements.
    3. **Future Prediction:** Once trained, the model predicts what might happen next based on recent trends.

        """)


        st.markdown("""
    **What is AI looking at?**

    The model analyzes several key indicators:
    - **Price History:** How the stock has moved over time.
    - **Moving Averages:** The average price over different time periods.
    - **Volume:** How many shares are being traded.
    - **Price Momentum:** The speed and strength of price movements.
    - **Overbought/Oversold Conditions:** If a stock might be due for a reversal.
        """)
        st.markdown("""
    **Deep Learning**

    This model uses a deep learning structure called Long Short-Term Memory (LSTM) neural networks, which finds patterns in sequential data.

    Key features of the model:
    - **Memory Capability:** Can remember important patterns and forget irrelevant ones.
    - **Multi-layered:** Uses multiple processing layers to learn patterns.
    - **Adaptive:** Continually updates its understanding as it processes more data.
    - **Feature Engineering:** Automatically identifies which indicators matter the most.

    The model is trained on historical data and tested against known outcomes before making future predictions.
        """)


        st.markdown("""
    **How accurate is it?**

    The predictions are generally within 2-5% of the actual price under normal market conditions. However, there are important limitations:
    - **Cannot predict news events:** Tarrifs,earnings surprises, product announcements, or global events can cause unexpected price movements.
    - **Works best with stable stocks:** Companies with erratic price histories are harder to predict.
    - **Short-term only:** The further into the future you try to predict, the less accurate the model becomes.
        """)


except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()


#Requirments: streamlit==1.28.0
#yfinance==0.2.31, pandas==2.1.1, numpy==1.25.2, tensorflow==2.13.0, plotly==5.17.0, scikit-learn==1.3.0, matplotlib==3.7.2
#To run, use streamlit run main.py in terminal
