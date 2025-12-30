"""
Super Peter Daytrader - Streamlit Dashboard

A comprehensive dashboard for analyzing trading performance,
training AI models, and making predictions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trading_bot import (
    DataLoader,
    FeatureEngineer,
    BehavioralCloner,
    BehavioralClonerConfig,
)

# Page configuration
st.set_page_config(
    page_title="Super Peter Daytrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📈 Super Peter Daytrader</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Behavioral Cloning Trading Bot")

# Sidebar
st.sidebar.title("🎛️ Control Panel")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "📁 Data Pipeline", "🤖 AI Training", "🔮 Predictions", "📚 Documentation"]
)

# Initialize session state
if "trades_df" not in st.session_state:
    st.session_state.trades_df = None
if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "training_set" not in st.session_state:
    st.session_state.training_set = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None


# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == "📊 Dashboard":
    st.header("📊 Trading Dashboard")

    st.markdown("""
    Welcome to the Super Peter Daytrader Dashboard! This application helps you:
    - 📁 **Load and analyze** trade orders from CSV files
    - 📊 **Visualize** trading performance and patterns
    - 🤖 **Train AI models** using behavioral cloning
    - 🔮 **Generate predictions** for future trades
    """)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Trades Loaded",
            value=len(st.session_state.trades_df) if st.session_state.trades_df is not None else 0
        )

    with col2:
        st.metric(
            label="Market Candles",
            value=len(st.session_state.market_df) if st.session_state.market_df is not None else 0
        )

    with col3:
        st.metric(
            label="Training Samples",
            value=len(st.session_state.training_set) if st.session_state.training_set is not None else 0
        )

    with col4:
        model_status = "Trained ✓" if st.session_state.trained_model is not None else "Not Trained"
        st.metric(
            label="AI Model",
            value=model_status
        )

    # Getting Started
    st.markdown("---")
    st.subheader("🚀 Getting Started")

    st.markdown("""
    1. **Upload Trade Orders**: Go to "📁 Data Pipeline" and upload your Orders.csv files
    2. **Train AI Model**: Navigate to "🤖 AI Training" to train a behavioral cloning model
    3. **Make Predictions**: Use "🔮 Predictions" to generate trading signals

    **Supported File Format:** CSV files with columns: `Fill Time, Product, B/S, Status, Exec Price`
    """)


# ============================================================================
# PAGE 2: DATA PIPELINE
# ============================================================================
elif page == "📁 Data Pipeline":
    st.header("📁 Data Pipeline")

    st.markdown("""
    Upload your trade order CSV files to begin analysis. You can upload **multiple files** at once.
    """)

    # File uploader - UPDATED TO ACCEPT MULTIPLE FILES
    uploaded_files = st.file_uploader(
        "Upload Orders CSV Files",
        type=["csv"],
        accept_multiple_files=True,  # ← KEY CHANGE: Enable multiple file upload
        help="Upload one or more CSV files containing your trade orders"
    )

    if uploaded_files:
        st.success(f"✓ Uploaded {len(uploaded_files)} file(s)")

        # Process all uploaded files
        all_trades = []

        for uploaded_file in uploaded_files:
            st.markdown(f"#### Processing: `{uploaded_file.name}`")

            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load trades
                loader = DataLoader()
                trades_df = loader.load_trades(temp_path)

                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", len(trades_df))
                with col2:
                    buy_count = (trades_df["target"] == 1).sum()
                    st.metric("Buy Orders", buy_count)
                with col3:
                    sell_count = (trades_df["target"] == -1).sum()
                    st.metric("Sell Orders", sell_count)

                # Add to collection
                all_trades.append(trades_df)

                # Display sample
                with st.expander(f"View Sample Data - {uploaded_file.name}"):
                    st.dataframe(trades_df.head(10))

                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        # Combine all trades
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
            st.session_state.trades_df = combined_trades

            st.markdown("---")
            st.subheader("📊 Combined Dataset Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", len(uploaded_files))
            with col2:
                st.metric("Total Trades", len(combined_trades))
            with col3:
                st.metric("Buy Orders", (combined_trades["target"] == 1).sum())
            with col4:
                st.metric("Sell Orders", (combined_trades["target"] == -1).sum())

            # Date range
            st.info(f"📅 Date Range: {combined_trades['timestamp'].min()} to {combined_trades['timestamp'].max()}")

            # Symbols
            symbols = combined_trades["symbol"].unique()
            st.info(f"📊 Symbols: {', '.join(symbols)}")

            # Fetch market data option
            st.markdown("---")
            st.subheader("📈 Fetch Market Data")

            symbol = st.selectbox("Select Symbol", symbols)

            if st.button("Fetch Market Data", type="primary"):
                with st.spinner("Fetching market data from yfinance..."):
                    try:
                        symbol_trades = combined_trades[combined_trades["symbol"] == symbol]
                        start_date = symbol_trades["timestamp"].min()
                        end_date = symbol_trades["timestamp"].max()

                        market_df = loader.fetch_market_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            interval="1m"
                        )

                        if market_df.empty:
                            st.warning("⚠️ No market data available. Using synthetic data for demonstration.")
                            # Create synthetic data
                            import numpy as np
                            timestamps = pd.date_range(start=start_date, end=end_date, freq="1min")
                            base_price = 21450.0
                            num_candles = len(timestamps)
                            np.random.seed(42)
                            price_changes = np.random.randn(num_candles) * 5
                            close_prices = base_price + np.cumsum(price_changes)

                            market_df = pd.DataFrame({
                                "timestamp": timestamps,
                                "open": close_prices + np.random.randn(num_candles) * 2,
                                "high": close_prices + np.abs(np.random.randn(num_candles)) * 3,
                                "low": close_prices - np.abs(np.random.randn(num_candles)) * 3,
                                "close": close_prices,
                                "volume": np.random.randint(100, 1000, num_candles),
                                "symbol": symbol,
                            })
                            market_df["high"] = market_df[["high", "close"]].max(axis=1)
                            market_df["low"] = market_df[["low", "close"]].min(axis=1)

                        st.session_state.market_df = market_df
                        st.success(f"✓ Fetched {len(market_df)} market candles")

                        # Create training set
                        training_set = loader.create_training_set(
                            symbol_trades,
                            market_df,
                            verbose=False
                        )
                        st.session_state.training_set = training_set

                        # Show chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=market_df["timestamp"],
                            open=market_df["open"],
                            high=market_df["high"],
                            low=market_df["low"],
                            close=market_df["close"]
                        )])
                        fig.update_layout(
                            title=f"{symbol} Price Chart",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error fetching market data: {str(e)}")


# ============================================================================
# PAGE 3: AI TRAINING
# ============================================================================
elif page == "🤖 AI Training":
    st.header("🤖 AI Model Training")

    if st.session_state.training_set is None:
        st.warning("⚠️ Please load trade data and fetch market data first (Data Pipeline page)")
    else:
        st.success(f"✓ Training set ready: {len(st.session_state.training_set)} samples")

        # Add features
        st.subheader("🔧 Feature Engineering")

        with st.spinner("Adding technical indicators..."):
            feature_engineer = FeatureEngineer()
            training_set = st.session_state.training_set.rename(columns={"timestamp": "date"})
            training_set = feature_engineer.transform(training_set)
            training_set["price_change"] = training_set["close"].pct_change()
            training_set["volume_change"] = training_set["volume"].pct_change()
            training_set["high_low_spread"] = (training_set["high"] - training_set["low"]) / training_set["close"]
            training_set = training_set.dropna().reset_index(drop=True)

        st.info(f"✓ Added technical indicators: RSI, EMA, Price Change, Volume Change, Spread")

        # Model configuration
        st.subheader("⚙️ Model Configuration")

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
        with col2:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random Seed", 0, 100, 42)

        # Train button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training AI model... This may take a moment."):
                try:
                    # Configure
                    config = BehavioralClonerConfig(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        test_size=test_size,
                        random_state=random_state,
                        class_weight="balanced"
                    )

                    # Initialize
                    cloner = BehavioralCloner(config)

                    # Prepare data
                    feature_columns = [
                        "close", "volume", "rsi", "ema",
                        "price_change", "volume_change", "high_low_spread"
                    ]
                    X = training_set[feature_columns]
                    y = training_set["target"]

                    # Train
                    metrics = cloner.train_model(X, y, verbose=False)

                    # Save to session state
                    st.session_state.trained_model = cloner

                    # Display results
                    st.success("✓ Model trained successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
                    with col2:
                        st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")

                    # Feature importance
                    st.subheader("📊 Feature Importance")
                    importance_df = pd.DataFrame({
                        "Feature": feature_columns,
                        "Importance": cloner.model.feature_importances_
                    }).sort_values("Importance", ascending=False)

                    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Confusion matrix
                    st.subheader("🎯 Confusion Matrix")
                    cm = metrics["confusion_matrix"]
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Sell", "Hold", "Buy"],
                        y=["Sell", "Hold", "Buy"],
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")


# ============================================================================
# PAGE 4: PREDICTIONS
# ============================================================================
elif page == "🔮 Predictions":
    st.header("🔮 Trading Predictions")

    if st.session_state.trained_model is None:
        st.warning("⚠️ Please train a model first (AI Training page)")
    else:
        st.success("✓ Model loaded and ready for predictions")

        if st.session_state.training_set is not None:
            # Get recent data
            training_set = st.session_state.training_set.copy()
            training_set = training_set.rename(columns={"timestamp": "date"})

            # Add features
            feature_engineer = FeatureEngineer()
            training_set = feature_engineer.transform(training_set)
            training_set["price_change"] = training_set["close"].pct_change()
            training_set["volume_change"] = training_set["volume"].pct_change()
            training_set["high_low_spread"] = (training_set["high"] - training_set["low"]) / training_set["close"]
            training_set = training_set.dropna()

            # Number of predictions
            n_predictions = st.slider("Number of Recent Candles to Predict", 5, 50, 10)

            if st.button("Generate Predictions", type="primary"):
                recent_data = training_set.tail(n_predictions).copy()

                feature_columns = [
                    "close", "volume", "rsi", "ema",
                    "price_change", "volume_change", "high_low_spread"
                ]

                X_recent = recent_data[feature_columns]
                predictions = st.session_state.trained_model.predict(X_recent)
                probabilities = st.session_state.trained_model.predict_proba(X_recent)

                # Add to dataframe
                recent_data["prediction"] = predictions
                recent_data["confidence"] = probabilities.max(axis=1) * 100

                # Map to labels
                action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
                recent_data["predicted_action"] = recent_data["prediction"].map(action_map)
                recent_data["actual_action"] = recent_data["target"].map(action_map)

                # Display results
                st.subheader("📋 Prediction Results")

                display_cols = ["date", "close", "rsi", "predicted_action", "actual_action", "confidence"]
                st.dataframe(
                    recent_data[display_cols].style.format({
                        "close": "{:.2f}",
                        "rsi": "{:.1f}",
                        "confidence": "{:.1f}%"
                    }),
                    use_container_width=True
                )

                # Accuracy
                correct = (recent_data["prediction"] == recent_data["target"]).sum()
                accuracy = correct / len(recent_data) * 100
                st.metric("Prediction Accuracy on Recent Data", f"{accuracy:.1f}%")

                # Distribution chart
                st.subheader("📊 Prediction Distribution")
                pred_counts = recent_data["predicted_action"].value_counts()
                fig = px.pie(values=pred_counts.values, names=pred_counts.index)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 5: DOCUMENTATION
# ============================================================================
elif page == "📚 Documentation":
    st.header("📚 Documentation")

    st.markdown("""
    ## About Super Peter Daytrader

    This application uses **Behavioral Cloning** to learn trading strategies from historical trade logs.

    ### Features

    - 📁 **Data Pipeline**: Load and process trade orders from CSV files
    - 🤖 **AI Training**: Train Random Forest models using behavioral cloning
    - 🔮 **Predictions**: Generate trading signals based on trained models
    - 📊 **Visualization**: Interactive charts and performance metrics

    ### CSV File Format

    Your Orders.csv should have these columns:

    | Column | Description | Example |
    |--------|-------------|---------|
    | Fill Time | Execution timestamp | "12/30/2024 09:31:15" |
    | Product | Trading symbol | "MNQ" |
    | B/S | Buy or Sell | " Buy" or " Sell" |
    | Status | Order status | " Filled" |
    | Exec Price | Execution price | 21450.25 |

    ### Supported Symbols

    - **MNQ**: Micro E-mini Nasdaq-100 Futures
    - **ES**: E-mini S&P 500 Futures
    - **GC**: Gold Futures
    - And more...

    ### How It Works

    1. **Load Trade Data**: Upload CSV files with your historical trades
    2. **Fetch Market Data**: Download 1-minute OHLCV data from yfinance
    3. **Merge Data**: Align trades with market candles
    4. **Add Features**: Calculate technical indicators (RSI, EMA, etc.)
    5. **Train Model**: Use Random Forest to learn patterns
    6. **Make Predictions**: Generate buy/sell/hold signals

    ### Requirements

    - Python 3.10+
    - Dependencies: pandas, yfinance, scikit-learn, streamlit, plotly

    ### Support

    For questions or issues, please refer to the documentation in the `docs/` folder.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        Super Peter Daytrader v1.0 | Built with Streamlit |
        <a href="https://github.com/anthropics/claude-code" target="_blank">Powered by Claude Code</a>
    </div>
    """,
    unsafe_allow_html=True
)
