def convert_to_timeframe(self, bars_data, target_timeframe='1h'):
    """Convert bars to different timeframe with proper timestamp handling"""
    print(f"🔄 Converting timeframe to: {target_timeframe}")
    print(f"🔄 Converting {len(bars_data)} bars to {target_timeframe}")

    if not bars_data:
        return []

    # Debug: Check original timestamp format
    print(f"🐛 First bar timestamp: {bars_data[0].datetime}")
    print(f"🐛 Last bar timestamp: {bars_data[-1].datetime}")
    print(f"🐛 Timestamp type: {type(bars_data[0].datetime)}")

    # Convert to pandas DataFrame for easier time resampling
    import pandas as pd

    # Create DataFrame with proper datetime index
    df_data = []
    for bar in bars_data:
        df_data.append({
            'datetime': bar.datetime,
            'open': bar.open_price,
            'high': bar.high_price,
            'low': bar.low_price,
            'close': bar.close_price,
            'volume': bar.volume
        })

    df = pd.DataFrame(df_data)

    # Debug: Check DataFrame info
    print(f"🐛 DataFrame shape: {df.shape}")
    print(f"🐛 DateTime range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"🐛 DataFrame info:")
    print(df.info())

    # Ensure datetime is properly formatted and timezone-aware
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Set datetime as index for resampling
    df = df.set_index('datetime')

    # Sort by datetime to ensure proper order
    df = df.sort_index()

    # Debug: Check index after conversion
    print(f"🐛 Index type: {type(df.index)}")
    print(f"🐛 Index range: {df.index.min()} to {df.index.max()}")
    print(f"🐛 Index frequency: {df.index.freq}")

    # Convert timeframe string to pandas frequency
    timeframe_map = {
        '1m': '1T',  # 1 minute
        '5m': '5T',  # 5 minutes
        '15m': '15T',  # 15 minutes
        '30m': '30T',  # 30 minutes
        '1h': '1H',  # 1 hour
        '4h': '4H',  # 4 hours
        '1d': '1D'  # 1 day
    }

    pandas_freq = timeframe_map.get(target_timeframe, '1H')
    print(f"🐛 Using pandas frequency: {pandas_freq}")

    # Resample using OHLC aggregation
    try:
        resampled = df.resample(pandas_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"🐛 Resampled shape: {resampled.shape}")
        print(f"🐛 Resampled head:")
        print(resampled.head())
        print(f"🐛 Resampled tail:")
        print(resampled.tail())

    except Exception as e:
        print(f"❌ Resampling error: {e}")
        return bars_data[:12]  # Fallback to first 12 bars

    # Convert back to bar objects
    converted_bars = []
    for timestamp, row in resampled.iterrows():
        # Create a new bar object (assuming you have a Bar class)
        bar = type(bars_data[0])()  # Create new instance of same type
        bar.datetime = timestamp.to_pydatetime() if hasattr(timestamp, 'to_pydatetime') else timestamp
        bar.open_price = row['open']
        bar.high_price = row['high']
        bar.low_price = row['low']
        bar.close_price = row['close']
        bar.volume = row['volume']
        converted_bars.append(bar)

    print(f"✅ BarGenerator converted to {len(converted_bars)} {target_timeframe} bars")

    # Debug: Check final result
    if converted_bars:
        print(f"🐛 Final first bar: {converted_bars[0].datetime}")
        print(f"🐛 Final last bar: {converted_bars[-1].datetime}")

    return converted_bars


def update_array_manager_with_proper_init(self, bars):
    """Update ArrayManager with proper initialization"""
    print(f"🔄 Updating ArrayManager with {len(bars)} bars")

    # Clear existing data
    self.am.close_array = np.array([])
    self.am.high_array = np.array([])
    self.am.low_array = np.array([])
    self.am.open_array = np.array([])
    self.am.volume_array = np.array([])

    # Add bars one by one to properly initialize
    for i, bar in enumerate(bars):
        # Update bar to ArrayManager
        self.am.update_bar(bar)

        # Debug first few iterations
        if i < 5:
            print(f"🐛 Bar {i}: {bar.datetime} - Close: {bar.close_price}")
            print(f"🐛 AM size: {self.am.size}, inited: {self.am.inited}")

    # Force initialization if we have enough data
    if len(bars) >= self.am.size:
        self.am.inited = True
        print(f"✅ ArrayManager force initialized with {len(bars)} bars")
    else:
        print(f"⚠️ Not enough bars for initialization. Need {self.am.size}, got {len(bars)}")

    print(f"✅ ArrayManager updated with {len(bars)} bars, initialized: {self.am.inited}")
    return self.am.inited