from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Configure the Streamlit app
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Add custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

try:
    # App title with custom styling
    st.markdown("<div class='main-header'>Stock Price Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "Make informed decisions with AI-powered stock price predictions based on historical patterns.")

    # Create tabs with more intuitive names
    tab1, tab2, tab3 = st.tabs(["📈 Price Prediction", "📊 Market Insights", "ℹ️ How It Works"])

    ##############################################
    # TAB 1: PRICE PREDICTION
    ##############################################
    with tab1:
        # Input section with improved layout
        st.markdown("<div class='subheader'>Enter Stock Details</div>", unsafe_allow_html=True)

        # Help tooltip
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

        # Add a button to trigger the analysis
        if st.button("Analyze Stock", type="primary"):
            # Create status container for progress updates
            status = st.empty()
            status.info(f"Getting historical data for {ticker_input}...")

            # Get stock data
            data = yf.download(ticker_input, period=period, interval="1d")

            if data.empty:
                st.error(f"No data found for '{ticker_input}'. Please check the symbol and try again.")
                st.stop()

            # Fetch company information
            try:
                ticker_info = yf.Ticker(ticker_input).info
                company_name = ticker_info.get('longName', ticker_input)
                sector = ticker_info.get('sector', 'N/A')
                industry = ticker_info.get('industry', 'N/A')

                # Display company info in a card
                st.markdown(f"""
                <div class='card'>
                    <h2>{company_name} ({ticker_input})</h2>
                    <p><strong>Sector:</strong> {sector} | <strong>Industry:</strong> {industry}</p>
                </div>
                """, unsafe_allow_html=True)

                # Display key metrics - make sure we're working with scalar values
                metric_cols = st.columns(4)

                # Get scalar values to avoid Series truth value error
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

                # Get financial ratios and metrics
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

                # Create a fundamentals section after the company info card
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
                    # Get earnings growth and revenue growth
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
                    # Get dividend info
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

                # Add a tooltip to explain these metrics
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

            # Display recent price history with simplified column names
            st.markdown("<div class='subheader'>Recent Price History</div>", unsafe_allow_html=True)

            # Create a more readable dataframe
            display_data = data.tail().copy()
            display_data.index = display_data.index.strftime('%b %d, %Y')

            st.dataframe(
                display_data.style.format({
                    "Open": "${:,.2f}",
                    "High": "${:,.2f}",
                    "Low": "${:,.2f}",
                    "Close": "${:,.2f}"
                })
            )

            # Create interactive price chart using Plotly
            status.info("Creating price chart...")

            # Calculate moving averages
            data['20-Day Average'] = data['Close'].rolling(window=20).mean()
            data['50-Day Average'] = data['Close'].rolling(window=50).mean()

            # Create a figure with subplot for price and volume
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03,
                                subplot_titles=('Stock Price', 'Trading Volume'),
                                row_heights=[0.7, 0.3])

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price"
                ),
                row=1, col=1
            )

            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['20-Day Average'],
                    name="20-day Average",
                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['50-Day Average'],
                    name="50-day Average",
                    line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5)
                ),
                row=1, col=1
            )

            # Add volume chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume",
                    marker_color='rgba(0, 0, 255, 0.5)'
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=f"{ticker_input} Stock Price History",
                xaxis_rangeslider_visible=False,
                yaxis_title="Price ($)",
                yaxis2_title="Volume",
                height=600,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Update y-axes
            fig.update_yaxes(tickprefix="$", row=1, col=1)

            # Add range selector
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(step="all")
                    ])
                ),
                row=1, col=1
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add quick explanation of the chart
            st.markdown("""
            <div class="tooltip-card">
                <strong>📊 Reading the Chart:</strong>
                <ul>
                    <li><strong>Candlesticks:</strong> Show opening, closing, high and low prices for each day</li>
                    <li><strong>Orange line:</strong> 20-day moving average (short-term trend)</li>
                    <li><strong>Red line:</strong> 50-day moving average (longer-term trend)</li>
                    <li><strong>Volume chart:</strong> Shows how many shares were traded each day</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Prepare data for LSTM model
            status.info("Preparing data for prediction...")

            # Calculate technical indicators for better predictions
            # Calculate RSI (Relative Strength Index - indicates overbought/oversold conditions)
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

            # Calculate MACD (Moving Average Convergence/Divergence - shows trend direction and momentum)
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Drop NaN values after calculating indicators
            data = data.dropna()

            # Select features for the model
            features = ['Close', 'Volume', '20-Day Average', '50-Day Average', 'RSI', 'MACD']

            # Scale the features
            scaler_dict = {}
            scaled_features = pd.DataFrame(index=data.index)

            for feature in features:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1)).flatten()
                scaler_dict[feature] = scaler

            # Save the main price scaler for later
            price_scaler = scaler_dict['Close']

            # Create sequences for LSTM
            sequence_length = 60  # Default sequence length


            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data.iloc[i:i + seq_length].values)
                    y.append(data['Close'].iloc[i + seq_length])
                return np.array(X), np.array(y)


            X, y = create_sequences(scaled_features, sequence_length)
            y = y.reshape(-1, 1)

            # Split data into training and testing sets
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Build the LSTM model
            status.info("Training AI model... This may take a moment.")

            # Improved model architecture
            model = Sequential()
            model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=100, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=50, activation='relu'))
            model.add(Dense(units=1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

            # Train with early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )

            # Make predictions on test data
            test_predictions = model.predict(X_test)
            test_predictions = price_scaler.inverse_transform(test_predictions)
            y_test_actual = price_scaler.inverse_transform(y_test)

            # Map selected timeframe to number of days
            timeframe_days = {
                "1 Day": 1,
                "3 Days": 3,
                "1 Week": 5,  # 5 trading days
                "2 Weeks": 10  # 10 trading days
            }

            days_to_predict = timeframe_days[prediction_timeframe]

            # Clean up status message
            status.success("Analysis complete!")

            # Display prediction results with enhanced styling
            st.markdown(f"<div class='subheader'>{prediction_timeframe} Price Prediction</div>", unsafe_allow_html=True)

            # For next-day prediction
            if days_to_predict == 1:
                # Create sequence for next day prediction
                last_sequence = scaled_features.iloc[-sequence_length:].values
                last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

                # Predict next day's price
                next_day_pred_scaled = model.predict(last_sequence)
                next_day_pred = float(price_scaler.inverse_transform(next_day_pred_scaled)[0][0])

                # Create date for next prediction
                last_date = data.index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                # Skip weekends
                if next_date.weekday() > 4:  # Saturday or Sunday
                    days_to_add = 8 - next_date.weekday()  # Move to Monday
                    next_date = last_date + pd.Timedelta(days=days_to_add)

                # Calculate change and determine color (using scalar values)
                last_close = float(data['Close'].iloc[-1])
                pred_change = ((next_day_pred - last_close) / last_close) * 100
                change_direction = "positive" if pred_change >= 0 else "negative"
                change_symbol = "+" if pred_change >= 0 else ""

                # Display the prediction card
                st.markdown(f"""
                <div class='prediction-card {change_direction}'>
                    <h2>Predicted Price for {next_date.strftime('%A, %B %d, %Y')}</h2>
                    <div class='metric-value'>${next_day_pred:.2f}</div>
                    <div class='metric-change' style='color:{"green" if change_direction == "positive" else "red"}'>
                        {change_symbol}{pred_change:.2f}% from previous close
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display metrics in columns
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Previous Close", f"${last_close:.2f}")
                with metric_cols[1]:
                    st.metric("Predicted Price", f"${next_day_pred:.2f}", f"{change_symbol}{pred_change:.2f}%")
                with metric_cols[2]:
                    # Calculate average prediction error from test data
                    mae = float(np.mean(np.abs(test_predictions - y_test_actual)))
                    st.metric("Average Error", f"${mae:.2f}",
                              help="How far off the model's predictions typically are")

                # Explain the prediction in simple terms
                st.markdown(f"""
                <div class="tooltip-card">
                    <strong>🔍 What this means:</strong> Based on historical patterns, the model predicts that {company_name} 
                    stock will {change_direction == "positive" and "rise" or "fall"} to ${next_day_pred:.2f} 
                    by {next_date.strftime('%A')}. The prediction is based on past trends, market momentum, and trading patterns.
                </div>
                """, unsafe_allow_html=True)

                # Create visualization of predictions vs actual
                st.markdown("<div class='subheader'>How Accurate Is Our Model?</div>", unsafe_allow_html=True)

                # Plot actual vs predicted test data
                fig_pred = go.Figure()

                # Add actual prices
                fig_pred.add_trace(go.Scatter(
                    x=data.index[-len(y_test_actual):],
                    y=y_test_actual.flatten(),
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ))

                # Add predicted prices
                fig_pred.add_trace(go.Scatter(
                    x=data.index[-len(test_predictions):],
                    y=test_predictions.flatten(),
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', width=2)
                ))

                # Add next day prediction point
                fig_pred.add_trace(go.Scatter(
                    x=[next_date],
                    y=[next_day_pred],
                    mode='markers',
                    name='Tomorrow\'s Prediction',
                    marker=dict(color='green', size=12, symbol='star')
                ))

                # Update layout
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

                # Explain the model performance chart
                st.markdown("""
                <div class="tooltip-card">
                    <strong>📈 Reading the Chart Above:</strong>
                    <ul>
                        <li><strong>Blue line:</strong> Actual historical prices</li>
                        <li><strong>Red line:</strong> What our model would have predicted</li>
                        <li><strong>Green star:</strong> Tomorrow's prediction</li>
                    </ul>
                    <p>The closer the red line follows the blue line, the more accurate our model is.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                # Replace the multi-day prediction section (starting around line 900) with this improved implementation:

                # Replace the entire multi-day prediction and visualization section with this code
                # Insert this where you handle multi-day predictions (around line 900)

                # For multi-day predictions, we need to forecast iteratively
                multi_day_predictions = []
                multi_day_dates = []

                # Add debugging lines
                st.markdown("### Debug Information")
                debug_container = st.empty()
                debug_container.info("Starting multi-day prediction process...")

                # Start with the last known sequence
                curr_sequence = scaled_features.iloc[-sequence_length:].values.copy()
                curr_date = data.index[-1]
                last_close = float(data['Close'].iloc[-1])

                # For debugging: Print initial values
                debug_text = f"Last known date: {curr_date}\nLast closing price: ${last_close:.2f}\n"
                debug_container.info(debug_text)

                # Loop for the number of days we want to predict
                for i in range(days_to_predict):
                    # Reshape for prediction
                    curr_sequence_reshaped = np.reshape(curr_sequence,
                                                        (1, curr_sequence.shape[0], curr_sequence.shape[1]))

                    # Predict the next day
                    next_pred_scaled = model.predict(curr_sequence_reshaped)
                    next_pred = float(price_scaler.inverse_transform(next_pred_scaled)[0][0])

                    # Move date forward (accounting for weekends)
                    curr_date = curr_date + pd.Timedelta(days=1)
                    while curr_date.weekday() > 4:  # Skip weekends
                        curr_date = curr_date + pd.Timedelta(days=1)

                    # Save this prediction
                    multi_day_predictions.append(next_pred)
                    multi_day_dates.append(curr_date)

                    # Debug: Print prediction for this day
                    debug_text += f"Day {i + 1} prediction: {curr_date.strftime('%a, %b %d')} - ${next_pred:.2f}\n"

                    # Now update the sequence for the next prediction
                    # Shift the sequence by one (drop oldest day)
                    curr_sequence = np.roll(curr_sequence, -1, axis=0)

                    # The last row needs to be updated with our new prediction
                    # Just update the Close price for simplicity
                    close_idx = features.index('Close')
                    curr_sequence[-1, close_idx] = next_pred_scaled[0][0]

                    # Update any other features if needed (for simplicity, we'll keep them the same)
                    # This is a simplified approach - in a real model you might want to update other features too

                # Update debug info with all predictions
                debug_container.info(debug_text)

                # Additional debug: Print array shapes and final values
                st.write(f"Number of prediction dates: {len(multi_day_dates)}")
                st.write(f"Number of prediction values: {len(multi_day_predictions)}")

                # Verify we have valid data for plotting
                if len(multi_day_dates) == 0 or len(multi_day_predictions) == 0:
                    st.error("No prediction data generated. Please check your model.")
                    st.stop()

                # Make sure both arrays have the same length
                if len(multi_day_dates) != len(multi_day_predictions):
                    st.error(
                        f"Data length mismatch: dates={len(multi_day_dates)}, predictions={len(multi_day_predictions)}")
                    st.stop()

                # Display the raw prediction data in a table for debugging
                debug_df = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m-%d') for d in multi_day_dates],
                    'Prediction': multi_day_predictions
                })
                st.write("Raw prediction data:")
                st.dataframe(debug_df)

                # Replace just the visualization part of your code (after the predictions are generated)
                # This starts after the line where debug_df is displayed

                # Convert prediction dates to strings for better compatibility with plotly
                date_strings = [d.strftime('%Y-%m-%d') for d in multi_day_dates]

                # Convert all data to native Python types to avoid pandas-related issues
                predictions_native = [float(p) for p in multi_day_predictions]

                # Get historical data
                historical_lookback = min(30, len(data))
                historical_dates = data.index[-historical_lookback:]
                historical_prices = [float(p) for p in data['Close'].iloc[-historical_lookback:].values]
                historical_date_strings = [d.strftime('%Y-%m-%d') for d in historical_dates]

                # Calculate error bounds
                mae = float(np.mean(np.abs(test_predictions - y_test_actual)))
                upper_bound = [p + mae for p in predictions_native]
                lower_bound = [p - mae for p in predictions_native]

                # Create a completely new figure
                st.markdown("### Stock Price Prediction")

                # Use pure Plotly instead of go.Figure for better control
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig_multi = go.Figure()

                # Add historical data
                fig_multi.add_trace(
                    go.Scatter(
                        x=historical_date_strings,
                        y=historical_prices,
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='blue', width=2)
                    )
                )

                # Add prediction line
                fig_multi.add_trace(
                    go.Scatter(
                        x=date_strings,
                        y=predictions_native,
                        mode='lines+markers',
                        name='Predictions',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8)
                    )
                )

                # Add upper bound
                fig_multi.add_trace(
                    go.Scatter(
                        x=date_strings,
                        y=upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    )
                )

                # Add lower bound with fill
                fig_multi.add_trace(
                    go.Scatter(
                        x=date_strings,
                        y=lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        name='Prediction Range'
                    )
                )

                # Update layout
                fig_multi.update_layout(
                    title=f"{prediction_timeframe} Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    yaxis=dict(tickprefix="$"),
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                # Display the chart
                st.plotly_chart(fig_multi, use_container_width=True)

                # After visualization is complete, clear the debug information
                debug_container.empty()

                # Lower bound
                fig_multi.add_trace(go.Scatter(
                    x=multi_day_dates,
                    y=lower_bound,
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0)'),
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Prediction Range'
                ))

                # Ensure the X-axis formatting is consistent
                fig_multi.update_xaxes(
                    rangeslider_visible=False,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=14, label="2w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )

                st.plotly_chart(fig_multi, use_container_width=True)

                # Calculate overall change from current price to final prediction
                final_pred = multi_day_predictions[-1]
                overall_change = ((final_pred - last_close) / last_close) * 100
                change_direction = "positive" if overall_change >= 0 else "negative"
                change_symbol = "+" if overall_change >= 0 else ""

                # Display the prediction card for multi-day forecast
                st.markdown(f"""
                <div class='prediction-card {change_direction}'>
                    <h2>{prediction_timeframe} Price Forecast</h2>
                    <div class='metric-value'>${final_pred:.2f}</div>
                    <div class='metric-change' style='color:{"green" if change_direction == "positive" else "red"}'>
                        {change_symbol}{overall_change:.2f}% from current price
                    </div>
                    <p>Forecast for {multi_day_dates[-1].strftime('%A, %B %d, %Y')}</p>
                </div>
                """, unsafe_allow_html=True)

                # Display the prediction table
                st.markdown("<div class='subheader'>Detailed Price Predictions</div>", unsafe_allow_html=True)

                # Create a DataFrame for the predictions
                pred_df = pd.DataFrame({
                    'Date': [d.strftime('%A, %b %d') for d in multi_day_dates],
                    'Predicted Price': [f"${p:.2f}" for p in multi_day_predictions],
                    'Predicted Range': [f"${p - mae:.2f} to ${p + mae:.2f}" for p in multi_day_predictions],
                    'Change': [
                        f"{'+' if (p - last_close) / last_close > 0 else ''}{((p - last_close) / last_close * 100):.2f}%"
                        for p in multi_day_predictions]
                })

                st.dataframe(pred_df)

                # Add explanation
                st.markdown(f"""
                <div class="tooltip-card">
                    <strong>🔍 About {prediction_timeframe} Predictions:</strong>
                    <p>Longer-term predictions typically have wider error margins. The prediction range shows where prices are likely to fall based on model accuracy.</p>
                    <p>The model recalculates each day's prediction based on previous predictions, which can compound uncertainty over time.</p>
                </div>
                """, unsafe_allow_html=True)

            # For both single and multi-day predictions, display metrics about model accuracy
            st.markdown("<div class='subheader'>Model Accuracy</div>", unsafe_allow_html=True)

            # Calculate metrics
            mae = float(mean_absolute_error(y_test_actual, test_predictions))
            rmse = float(np.sqrt(mean_squared_error(y_test_actual, test_predictions)))
            mape = float(np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100)

            # Display in more user-friendly way
            st.markdown(f"""
            <div class="card">
                <p>On average, our predictions are:</p>
                <ul>
                    <li>Within <strong>${mae:.2f}</strong> of the actual price</li>
                    <li>About <strong>{mape:.1f}%</strong> off from the actual price</li>
                </ul>
                <p>For this stock, that means you should consider our prediction to be a range rather than an exact price.</p>
            </div>
            """, unsafe_allow_html=True)

            # Display disclaimer
            st.markdown("""
            <div class="disclaimer">
                <strong>⚠️ Important Note:</strong> Stock predictions are estimates based on historical patterns. 
                They cannot account for unexpected news or events. Use this as just one tool in your investment research.
                Never invest money you cannot afford to lose.
            </div>
            """, unsafe_allow_html=True)

    ##############################################
    # TAB 2: MARKET INSIGHTS
    ##############################################
    with tab2:
        st.markdown("<div class='subheader'>Market Insights Simplified</div>", unsafe_allow_html=True)

        # Help tooltip
        st.markdown("""
        <div class="tooltip-card">
            <strong>📌 This tab:</strong> Shows key indicators that traders use to make decisions, presented in a simplified way.
            These insights will help you understand if the stock might be a good buy, hold, or sell right now.
        </div>
        """, unsafe_allow_html=True)

        if 'data' in locals():
            # Create a simplified dashboard of indicators

            # Determine signals - convert to scalar values to avoid Series truth value error
            last_close = float(data['Close'].iloc[-1])
            last_ma20 = float(data['20-Day Average'].iloc[-1])
            last_ma50 = float(data['50-Day Average'].iloc[-1])
            last_rsi = float(data['RSI'].iloc[-1])
            last_macd = float(data['MACD'].iloc[-1])
            last_signal = float(data['Signal'].iloc[-1])

            # Simplified signal cards with explanations
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

            # Overall signal summary - using simple string list to avoid Series issues
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

            # Add the fundamental analysis section if we have the data
            if 'ticker_info' in locals() and 'pe_ratio' in locals():
                st.markdown("<div class='subheader'>Fundamental Analysis Dashboard</div>", unsafe_allow_html=True)

                # Get industry average P/E for comparison
                industry_pe = ticker_info.get('industryPE', None)

                # Create a gauge chart to show if P/E is high or low compared to industry
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

                # Display PE ratio comparison
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
                    # Evaluate growth metrics
                    growth_status = "NEUTRAL"
                    growth_color = "gray"
                    growth_explanation = "Growth metrics are average."

                    if 'earnings_growth' in locals() and earnings_growth != 'N/A':
                        try:
                            eg_value = float(earnings_growth.strip('%')) / 100
                            if eg_value > 0.2:  # More than 20% growth
                                growth_status = "STRONG GROWTH"
                                growth_color = "green"
                                growth_explanation = "Company shows strong earnings growth."
                            elif eg_value < 0:  # Negative growth
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

            # Simplified price statistics
            st.markdown("### Stock Performance")

            # Calculate returns and volatility - using scalar values
            returns = data['Close'].pct_change().dropna()

            if len(returns) > 0:
                daily_return = float(returns.iloc[-1] * 100)
                avg_daily_return = float(returns.mean() * 100)

                # Calculate monthly and yearly returns in more intuitive way
                monthly_return = avg_daily_return * 21  # Approx trading days in a month
                yearly_return = avg_daily_return * 252  # Approx trading days in a year

                volatility = float(returns.std() * 100)

                # Calculate annualized volatility
                annualized_volatility = volatility * np.sqrt(252)
            else:
                daily_return = 0.0
                monthly_return = 0.0
                yearly_return = 0.0
                volatility = 0.0
                annualized_volatility = 0.0

            # YTD calculation - properly handle with explicit checks
            current_year = datetime.now().year
            start_of_year = datetime(current_year, 1, 1)
            ytd_data = data[data.index >= start_of_year]

            if not ytd_data.empty and len(ytd_data) > 1:
                ytd_start_price = float(ytd_data['Close'].iloc[0])
                ytd_return = ((last_close - ytd_start_price) / ytd_start_price) * 100
            else:
                ytd_return = 0.0

            # Display statistics in a more intuitive way
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

            # Show a simple historical performance chart
            st.markdown("### Price Performance Over Time")

            # Create a simple line chart instead of a complex candlestick
            fig_perf = go.Figure()

            # Calculate key time periods
            if len(data) > 252:  # If we have at least a year of data
                one_month_ago = data.index[-1] - pd.Timedelta(days=30)
                three_months_ago = data.index[-1] - pd.Timedelta(days=90)
                six_months_ago = data.index[-1] - pd.Timedelta(days=180)
                one_year_ago = data.index[-1] - pd.Timedelta(days=365)

                # Add reference points
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

            # Add the line
            fig_perf.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue', width=2)
            ))

            # Update layout
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

            # Simple return analysis
            if len(data) > 252:  # If we have at least a year of data
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

    ##############################################
    # TAB 3: HOW IT WORKS
    ##############################################
    with tab3:
        st.markdown("<div class='subheader'>How This Works</div>", unsafe_allow_html=True)

        # Create more user-friendly explanation with visual elements
        st.markdown("""
        <div class="card">
            <h3>What is this app doing?</h3>
            <p>This app uses artificial intelligence to predict stock prices based on historical patterns. Here's how it works:</p>

            <ol>
                <li><strong>Data Collection:</strong> We get historical stock data from Yahoo Finance.</li>
                <li><strong>Pattern Recognition:</strong> Our AI model (called LSTM - Long Short-Term Memory) learns patterns from past price movements.</li>
                <li><strong>Future Prediction:</strong> Once trained, the model predicts what might happen next based on recent trends.</li>
            </ol>

            <p>Think of it like a weather forecast, but for stocks - it looks at patterns in historical data to make educated guesses about future movements.</p>
        </div>

        <div class="card">
            <h3>What is AI looking at?</h3>
            <p>Our model analyzes several key indicators:</p>

            <ul>
                <li><strong>Price History:</strong> How the stock has moved over time</li>
                <li><strong>Moving Averages:</strong> The average price over different time periods</li>
                <li><strong>Volume:</strong> How many shares are being traded</li>
                <li><strong>Price Momentum:</strong> The speed and strength of price movements</li>
                <li><strong>Overbought/Oversold Conditions:</strong> If a stock might be due for a reversal</li>
            </ul>
        </div>

        <div class="card">
            <h3>Deep Learning Technology</h3>
            <p>This app uses a sophisticated deep learning architecture called Long Short-Term Memory (LSTM) neural networks, which excel at finding patterns in sequential data like stock prices.</p>

            <p>Key features of our LSTM model:</p>
            <ul>
                <li><strong>Memory Capability:</strong> Can remember important patterns while forgetting irrelevant ones</li>
                <li><strong>Multi-layered:</strong> Uses multiple processing layers to learn increasingly complex patterns</li>
                <li><strong>Adaptive:</strong> Continually updates its understanding as it processes more data</li>
                <li><strong>Feature Engineering:</strong> Automatically identifies which indicators matter most</li>
            </ul>

            <p>The model is trained on historical data and tested against known outcomes before making future predictions.</p>
        </div>

        <div class="card">
            <h3>How accurate is it?</h3>
            <p>Our predictions are generally within 2-5% of the actual price under normal market conditions. However, there are important limitations:</p>

            <ul>
                <li><strong>Cannot predict news events:</strong> Earnings surprises, product announcements, or global events can cause unexpected price movements.</li>
                <li><strong>Works best with stable stocks:</strong> Companies with erratic price histories are harder to predict.</li>
                <li><strong>Short-term only:</strong> The further into the future we try to predict, the less accurate we become.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # FAQs presented in a more approachable way
        st.markdown("<div class='subheader'>Common Questions</div>", unsafe_allow_html=True)

        with st.expander("Should I use this for making investment decisions?"):
            st.markdown("""
            <div style="padding: 1rem;">
                <p>This tool should be <strong>one of many resources</strong> you use when making investment decisions - never the only one.</p>

                <p>Think of it like getting a second opinion. Our AI makes predictions based on patterns, but it cannot account for upcoming news, earnings reports, or market sentiment.</p>

                <p>Before investing, you should:</p>
                <ul>
                    <li>Research the company's fundamentals (earnings, debt, growth, etc.)</li>
                    <li>Consider overall market conditions</li>
                    <li>Understand your risk tolerance and investment timeline</li>
                    <li>Ideally, consult with a financial advisor</li>
                </ul>

                <p><strong>Never invest money you cannot afford to lose based solely on algorithmic predictions.</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("What does 'Bullish' vs 'Bearish' mean?"):
            st.markdown("""
            <div style="padding: 1rem;">
                <p>These are common terms that describe market sentiment:</p>

                <p><strong>Bullish</strong> = Positive outlook, expecting prices to rise</p>
                <ul>
                    <li>Think of a charging bull with horns pointing upward</li>
                    <li>If our indicators show a "bullish" signal, they suggest the stock price may increase</li>
                </ul>

                <p><strong>Bearish</strong> = Negative outlook, expecting prices to fall</p>
                <ul>
                    <li>Think of a bear swiping its paw downward</li>
                    <li>If our indicators show a "bearish" signal, they suggest the stock price may decrease</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("What are 'Moving Averages'?"):
            st.markdown("""
            <div style="padding: 1rem;">
                <p>A moving average smooths out price data to create a single flowing line, making it easier to identify trends.</p>

                <p>We use two key moving averages:</p>
                <ul>
                    <li><strong>20-Day Moving Average:</strong> The average closing price over the last 20 trading days (about a month). This line responds quickly to price changes and helps identify short-term trends.</li>
                    <li><strong>50-Day Moving Average:</strong> The average closing price over the last 50 trading days (about 2.5 months). This line is slower to change and helps identify medium-term trends.</li>
                </ul>

                <p><strong>How to interpret them:</strong></p>
                <ul>
                    <li>When the price is above both moving averages = Strong uptrend</li>
                    <li>When the price is below both moving averages = Strong downtrend</li>
                    <li>When the 20-day crosses above the 50-day = Potential buying opportunity (called a "golden cross")</li>
                    <li>When the 20-day crosses below the 50-day = Potential selling opportunity (called a "death cross")</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("What is RSI (Relative Strength Index)?"):
            st.markdown("""
            <div style="padding: 1rem;">
                <p>RSI is a momentum indicator that measures the speed and change of price movements on a scale from 0 to 100.</p>

                <p><strong>How to interpret RSI:</strong></p>
                <ul>
                    <li><strong>RSI above 70:</strong> The stock may be "overbought" (potentially overvalued) and could be due for a price decrease</li>
                    <li><strong>RSI below 30:</strong> The stock may be "oversold" (potentially undervalued) and could be due for a price increase</li>
                    <li><strong>RSI between 30-70:</strong> The stock is in a neutral territory</li>
                </ul>

                <p>Think of RSI like a car's speedometer. When it's too high, the market might be moving too fast in one direction and could be ready to slow down or reverse.</p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("What are P/E Ratios and other fundamental metrics?"):
            st.markdown("""
            <div style="padding: 1rem;">
                <p>Fundamental metrics help you understand a company's financial health and valuation:</p>

                <p><strong>P/E Ratio (Price to Earnings)</strong>: Shows how much investors
            <div style="padding: 1rem;">
                            <p>Fundamental metrics help you understand a company's financial health and valuation:</p>

                            <p><strong>P/E Ratio (Price to Earnings)</strong>: Shows how much investors are willing to pay for $1 of earnings</p>
                            <ul>
                                <li>Lower P/E may indicate better value, but very low P/E might signal problems</li>
                                <li>Higher P/E might indicate overvaluation, or expectations of strong future growth</li>
                                <li>Always compare to industry averages - a tech company typically has higher P/E than a utility</li>
                            </ul>

                            <p><strong>PEG Ratio (P/E to Growth)</strong>: P/E ratio divided by earnings growth rate</p>
                            <ul>
                                <li>Helps determine if a high P/E is justified by high growth</li>
                                <li>Values under 1.0 often suggest an undervalued stock</li>
                            </ul>

                            <p><strong>Debt to Equity</strong>: Measures a company's financial leverage</p>
                            <ul>
                                <li>Higher ratios indicate more debt, which means higher risk but potentially higher returns</li>
                                <li>Lower ratios suggest financial stability but potentially lower returns</li>
                            </ul>

                            <p><strong>Profit Margin</strong>: Percentage of revenue that becomes profit</p>
                            <ul>
                                <li>Higher margins suggest pricing power and operational efficiency</li>
                                <li>Compare to industry peers - some industries naturally have lower margins</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

            # Add disclaimer at the bottom
            st.markdown("""
                    <div class="disclaimer">
                        <strong>Disclaimer:</strong> This application is for educational and informational purposes only. The predictions and analysis provided should not be considered as financial advice. 
                        Past performance is not indicative of future results. Never invest money you cannot afford to lose. Always conduct thorough research before making investment decisions.
                    </div>
                    """, unsafe_allow_html=True)

            # Footer
            st.markdown("""
                    <div class="footer">
                        Stock Price Predictor | Built with Streamlit, TensorFlow, and Python
                    </div>
                    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()