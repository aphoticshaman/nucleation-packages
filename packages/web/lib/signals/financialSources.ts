/**
 * FINANCIAL DATA SOURCES
 *
 * Real-time market data APIs for stock, bond, forex, and crypto analysis.
 * All sources categorized by:
 * - Cost tier (free, freemium, paid)
 * - Data type (stocks, bonds, forex, crypto, macro)
 * - Update frequency (real-time, delayed, daily)
 * - Authentication requirements
 */

export interface FinancialSource {
  id: string;
  name: string;
  description: string;
  endpoint: string;
  apiKeyEnv: string | null; // null = no auth required
  dataTypes: DataType[];
  coverage: string[];
  rateLimit: {
    requests: number;
    window: 'second' | 'minute' | 'hour' | 'day' | 'month';
  };
  latency: 'realtime' | '15min' | 'hourly' | 'daily';
  costTier: 'free' | 'freemium' | 'paid';
  costPerMonth: number;
  documentation: string;
  enabled: boolean;
  priority: 1 | 2 | 3;
}

type DataType =
  | 'stocks'
  | 'etfs'
  | 'bonds'
  | 'forex'
  | 'crypto'
  | 'commodities'
  | 'options'
  | 'futures'
  | 'macro'
  | 'sentiment'
  | 'fundamentals'
  | 'news';

// ============================================
// FREE TIER SOURCES (No payment required)
// ============================================

export const FREE_FINANCIAL_SOURCES: FinancialSource[] = [
  {
    id: 'alpha_vantage',
    name: 'Alpha Vantage',
    description: 'Stocks, forex, crypto, fundamentals. Best free stock API.',
    endpoint: 'https://www.alphavantage.co/query',
    apiKeyEnv: 'ALPHA_VANTAGE_API_KEY',
    dataTypes: ['stocks', 'forex', 'crypto', 'fundamentals', 'macro'],
    coverage: ['global', 'us_stocks', 'forex_pairs', 'crypto'],
    rateLimit: { requests: 25, window: 'day' }, // Free tier
    latency: 'realtime',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://www.alphavantage.co/documentation/',
    enabled: true,
    priority: 1,
  },
  {
    id: 'finnhub',
    name: 'Finnhub',
    description: 'Real-time stock data, company news, financials.',
    endpoint: 'https://finnhub.io/api/v1',
    apiKeyEnv: 'FINNHUB_API_KEY',
    dataTypes: ['stocks', 'forex', 'crypto', 'news', 'sentiment'],
    coverage: ['us_stocks', 'global_stocks', 'forex', 'crypto'],
    rateLimit: { requests: 60, window: 'minute' }, // Free tier
    latency: 'realtime',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://finnhub.io/docs/api',
    enabled: true,
    priority: 1,
  },
  {
    id: 'polygon',
    name: 'Polygon.io',
    description: 'Stocks, options, forex, crypto. Excellent free tier.',
    endpoint: 'https://api.polygon.io',
    apiKeyEnv: 'POLYGON_API_KEY',
    dataTypes: ['stocks', 'options', 'forex', 'crypto'],
    coverage: ['us_stocks', 'us_options', 'forex', 'crypto'],
    rateLimit: { requests: 5, window: 'minute' }, // Free tier
    latency: '15min', // Free tier is delayed
    costTier: 'freemium',
    costPerMonth: 0,
    documentation: 'https://polygon.io/docs',
    enabled: true,
    priority: 2,
  },
  {
    id: 'twelve_data',
    name: 'Twelve Data',
    description: 'Stocks, forex, crypto, ETFs. Good technical indicators.',
    endpoint: 'https://api.twelvedata.com',
    apiKeyEnv: 'TWELVE_DATA_API_KEY',
    dataTypes: ['stocks', 'etfs', 'forex', 'crypto', 'fundamentals'],
    coverage: ['global_stocks', 'etfs', 'forex', 'crypto'],
    rateLimit: { requests: 800, window: 'day' }, // Free tier
    latency: 'realtime',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://twelvedata.com/docs',
    enabled: true,
    priority: 2,
  },
  {
    id: 'yahoo_finance',
    name: 'Yahoo Finance (Unofficial)',
    description: 'Comprehensive stock data via yfinance. Free, no key.',
    endpoint: 'https://query1.finance.yahoo.com/v8/finance',
    apiKeyEnv: null, // No API key needed
    dataTypes: ['stocks', 'etfs', 'bonds', 'forex', 'crypto', 'fundamentals'],
    coverage: ['global', 'comprehensive'],
    rateLimit: { requests: 2000, window: 'hour' }, // Unofficial limit
    latency: '15min',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://github.com/ranaroussi/yfinance',
    enabled: true,
    priority: 1,
  },
  {
    id: 'fred',
    name: 'Federal Reserve Economic Data',
    description: 'US economic indicators, rates, yields. Official Fed data.',
    endpoint: 'https://api.stlouisfed.org/fred',
    apiKeyEnv: 'FRED_API_KEY',
    dataTypes: ['macro', 'bonds'],
    coverage: ['usa', 'treasury_yields', 'economic_indicators'],
    rateLimit: { requests: 120, window: 'minute' },
    latency: 'daily',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://fred.stlouisfed.org/docs/api/fred/',
    enabled: true,
    priority: 1,
  },
  {
    id: 'ecb',
    name: 'European Central Bank',
    description: 'EUR exchange rates, EU economic data.',
    endpoint: 'https://data-api.ecb.europa.eu',
    apiKeyEnv: null,
    dataTypes: ['forex', 'macro'],
    coverage: ['eurozone', 'forex', 'monetary_policy'],
    rateLimit: { requests: 100, window: 'minute' },
    latency: 'daily',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://data.ecb.europa.eu/help/api',
    enabled: true,
    priority: 2,
  },
  {
    id: 'coingecko',
    name: 'CoinGecko',
    description: 'Comprehensive crypto data. Market cap, volume, prices.',
    endpoint: 'https://api.coingecko.com/api/v3',
    apiKeyEnv: null,
    dataTypes: ['crypto'],
    coverage: ['crypto', '13000+_coins', 'exchanges'],
    rateLimit: { requests: 50, window: 'minute' },
    latency: 'realtime',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://www.coingecko.com/en/api/documentation',
    enabled: true,
    priority: 1,
  },
  {
    id: 'binance',
    name: 'Binance Public API',
    description: 'Crypto trading data, order books, trades.',
    endpoint: 'https://api.binance.com/api/v3',
    apiKeyEnv: null, // Public endpoints don't need key
    dataTypes: ['crypto'],
    coverage: ['crypto', 'trading_pairs', 'order_books'],
    rateLimit: { requests: 1200, window: 'minute' },
    latency: 'realtime',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://binance-docs.github.io/apidocs/',
    enabled: true,
    priority: 2,
  },
  {
    id: 'open_exchange_rates',
    name: 'Open Exchange Rates',
    description: 'Currency exchange rates. 200 currencies.',
    endpoint: 'https://openexchangerates.org/api',
    apiKeyEnv: 'OPEN_EXCHANGE_RATES_APP_ID',
    dataTypes: ['forex'],
    coverage: ['forex', '200_currencies'],
    rateLimit: { requests: 1000, window: 'month' }, // Free tier
    latency: 'hourly',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://docs.openexchangerates.org/',
    enabled: true,
    priority: 2,
  },
  {
    id: 'exchangerate_api',
    name: 'ExchangeRate-API',
    description: 'Simple forex rates. 1500 free requests/month.',
    endpoint: 'https://v6.exchangerate-api.com/v6',
    apiKeyEnv: 'EXCHANGERATE_API_KEY',
    dataTypes: ['forex'],
    coverage: ['forex', '161_currencies'],
    rateLimit: { requests: 1500, window: 'month' },
    latency: 'daily',
    costTier: 'free',
    costPerMonth: 0,
    documentation: 'https://www.exchangerate-api.com/docs',
    enabled: true,
    priority: 3,
  },
];

// ============================================
// PREMIUM SOURCES (Paid tiers available)
// ============================================

export const PREMIUM_FINANCIAL_SOURCES: FinancialSource[] = [
  {
    id: 'iex_cloud',
    name: 'IEX Cloud',
    description: 'Professional-grade stock data. Excellent fundamentals.',
    endpoint: 'https://cloud.iexapis.com/stable',
    apiKeyEnv: 'IEX_CLOUD_API_KEY',
    dataTypes: ['stocks', 'etfs', 'fundamentals', 'news'],
    coverage: ['us_stocks', 'fundamentals', 'insider_trading'],
    rateLimit: { requests: 50000, window: 'month' }, // Free tier
    latency: 'realtime',
    costTier: 'freemium',
    costPerMonth: 0, // Free tier, paid starts at $9/mo
    documentation: 'https://iexcloud.io/docs/api/',
    enabled: false,
    priority: 1,
  },
  {
    id: 'tradier',
    name: 'Tradier',
    description: 'Real-time quotes, options chains, historical data.',
    endpoint: 'https://api.tradier.com/v1',
    apiKeyEnv: 'TRADIER_API_KEY',
    dataTypes: ['stocks', 'options', 'etfs'],
    coverage: ['us_stocks', 'options', 'etfs'],
    rateLimit: { requests: 120, window: 'minute' },
    latency: 'realtime',
    costTier: 'freemium',
    costPerMonth: 0, // Sandbox is free
    documentation: 'https://documentation.tradier.com/',
    enabled: false,
    priority: 2,
  },
  {
    id: 'quandl',
    name: 'Nasdaq Data Link (Quandl)',
    description: 'Alternative data, futures, commodities.',
    endpoint: 'https://data.nasdaq.com/api/v3',
    apiKeyEnv: 'QUANDL_API_KEY',
    dataTypes: ['stocks', 'futures', 'commodities', 'macro'],
    coverage: ['alternative_data', 'futures', 'commodities'],
    rateLimit: { requests: 50, window: 'day' }, // Free tier
    latency: 'daily',
    costTier: 'freemium',
    costPerMonth: 0,
    documentation: 'https://docs.data.nasdaq.com/',
    enabled: false,
    priority: 2,
  },
  {
    id: 'marketstack',
    name: 'Marketstack',
    description: 'Real-time stock data. 170+ exchanges.',
    endpoint: 'http://api.marketstack.com/v1',
    apiKeyEnv: 'MARKETSTACK_API_KEY',
    dataTypes: ['stocks', 'etfs'],
    coverage: ['global_stocks', '170_exchanges'],
    rateLimit: { requests: 100, window: 'month' }, // Free tier
    latency: 'realtime',
    costTier: 'freemium',
    costPerMonth: 0, // Paid starts at $9.99/mo
    documentation: 'https://marketstack.com/documentation',
    enabled: false,
    priority: 3,
  },
];

// ============================================
// ALL SOURCES COMBINED
// ============================================

export const ALL_FINANCIAL_SOURCES: FinancialSource[] = [
  ...FREE_FINANCIAL_SOURCES,
  ...PREMIUM_FINANCIAL_SOURCES,
];

// ============================================
// KEY MARKET INDICATORS
// ============================================

export const MARKET_INDICATORS = {
  // US Indices
  us_indices: [
    { symbol: 'SPY', name: 'S&P 500 ETF', type: 'index' },
    { symbol: 'QQQ', name: 'NASDAQ-100 ETF', type: 'index' },
    { symbol: 'DIA', name: 'Dow Jones ETF', type: 'index' },
    { symbol: 'IWM', name: 'Russell 2000 ETF', type: 'index' },
    { symbol: 'VIX', name: 'Volatility Index', type: 'volatility' },
  ],

  // Sector ETFs
  sectors: [
    { symbol: 'XLK', name: 'Technology', type: 'sector' },
    { symbol: 'XLF', name: 'Financials', type: 'sector' },
    { symbol: 'XLE', name: 'Energy', type: 'sector' },
    { symbol: 'XLV', name: 'Healthcare', type: 'sector' },
    { symbol: 'XLI', name: 'Industrials', type: 'sector' },
    { symbol: 'XLP', name: 'Consumer Staples', type: 'sector' },
    { symbol: 'XLY', name: 'Consumer Discretionary', type: 'sector' },
    { symbol: 'XLU', name: 'Utilities', type: 'sector' },
    { symbol: 'XLB', name: 'Materials', type: 'sector' },
    { symbol: 'XLRE', name: 'Real Estate', type: 'sector' },
    { symbol: 'XLC', name: 'Communication Services', type: 'sector' },
  ],

  // Treasury Yields (FRED series)
  treasury: [
    { symbol: 'DGS1MO', name: '1-Month Treasury', type: 'yield' },
    { symbol: 'DGS3MO', name: '3-Month Treasury', type: 'yield' },
    { symbol: 'DGS6MO', name: '6-Month Treasury', type: 'yield' },
    { symbol: 'DGS1', name: '1-Year Treasury', type: 'yield' },
    { symbol: 'DGS2', name: '2-Year Treasury', type: 'yield' },
    { symbol: 'DGS5', name: '5-Year Treasury', type: 'yield' },
    { symbol: 'DGS10', name: '10-Year Treasury', type: 'yield' },
    { symbol: 'DGS30', name: '30-Year Treasury', type: 'yield' },
    { symbol: 'T10Y2Y', name: '10Y-2Y Spread (Inversion)', type: 'spread' },
  ],

  // Commodities
  commodities: [
    { symbol: 'GC=F', name: 'Gold Futures', type: 'commodity' },
    { symbol: 'SI=F', name: 'Silver Futures', type: 'commodity' },
    { symbol: 'CL=F', name: 'Crude Oil WTI', type: 'commodity' },
    { symbol: 'BZ=F', name: 'Brent Crude', type: 'commodity' },
    { symbol: 'NG=F', name: 'Natural Gas', type: 'commodity' },
    { symbol: 'HG=F', name: 'Copper', type: 'commodity' },
    { symbol: 'ZC=F', name: 'Corn', type: 'commodity' },
    { symbol: 'ZW=F', name: 'Wheat', type: 'commodity' },
    { symbol: 'ZS=F', name: 'Soybeans', type: 'commodity' },
  ],

  // Major Forex Pairs
  forex: [
    { symbol: 'EURUSD', name: 'EUR/USD', type: 'forex' },
    { symbol: 'GBPUSD', name: 'GBP/USD', type: 'forex' },
    { symbol: 'USDJPY', name: 'USD/JPY', type: 'forex' },
    { symbol: 'USDCHF', name: 'USD/CHF', type: 'forex' },
    { symbol: 'AUDUSD', name: 'AUD/USD', type: 'forex' },
    { symbol: 'USDCAD', name: 'USD/CAD', type: 'forex' },
    { symbol: 'NZDUSD', name: 'NZD/USD', type: 'forex' },
    { symbol: 'DXY', name: 'US Dollar Index', type: 'forex' },
  ],

  // Crypto
  crypto: [
    { symbol: 'BTC', name: 'Bitcoin', type: 'crypto' },
    { symbol: 'ETH', name: 'Ethereum', type: 'crypto' },
    { symbol: 'SOL', name: 'Solana', type: 'crypto' },
    { symbol: 'BNB', name: 'Binance Coin', type: 'crypto' },
    { symbol: 'XRP', name: 'Ripple', type: 'crypto' },
    { symbol: 'ADA', name: 'Cardano', type: 'crypto' },
    { symbol: 'DOGE', name: 'Dogecoin', type: 'crypto' },
    { symbol: 'DOT', name: 'Polkadot', type: 'crypto' },
  ],

  // Economic Indicators (FRED)
  economic: [
    { symbol: 'UNRATE', name: 'Unemployment Rate', type: 'macro' },
    { symbol: 'CPIAUCSL', name: 'CPI (Inflation)', type: 'macro' },
    { symbol: 'FEDFUNDS', name: 'Fed Funds Rate', type: 'macro' },
    { symbol: 'GDP', name: 'US GDP', type: 'macro' },
    { symbol: 'M2SL', name: 'M2 Money Supply', type: 'macro' },
    { symbol: 'PAYEMS', name: 'Nonfarm Payrolls', type: 'macro' },
    { symbol: 'UMCSENT', name: 'Consumer Sentiment', type: 'macro' },
    { symbol: 'HOUST', name: 'Housing Starts', type: 'macro' },
  ],
};

// ============================================
// DATA QUALITY TRACKING
// ============================================

export interface DataFreshness {
  sourceId: string;
  lastFetchTime: Date;
  lastSuccessTime: Date | null;
  lastError: string | null;
  dataAge: number; // seconds since last successful fetch
  status: 'fresh' | 'stale' | 'error' | 'unknown';
  recordCount: number;
  stalenessThreshold: number; // seconds before considered stale
}

export const STALENESS_THRESHOLDS: Record<string, number> = {
  // Real-time sources (stale after 5 min)
  realtime: 5 * 60,

  // 15-min delayed (stale after 30 min)
  delayed: 30 * 60,

  // Hourly updates (stale after 2 hours)
  hourly: 2 * 60 * 60,

  // Daily updates (stale after 36 hours)
  daily: 36 * 60 * 60,

  // Weekly updates (stale after 10 days)
  weekly: 10 * 24 * 60 * 60,
};

/**
 * Calculate data freshness status
 */
export function calculateFreshness(
  lastSuccess: Date | null,
  latencyType: FinancialSource['latency']
): DataFreshness['status'] {
  if (!lastSuccess) return 'unknown';

  const age = (Date.now() - lastSuccess.getTime()) / 1000;

  let threshold: number;
  switch (latencyType) {
    case 'realtime':
      threshold = STALENESS_THRESHOLDS.realtime;
      break;
    case '15min':
      threshold = STALENESS_THRESHOLDS.delayed;
      break;
    case 'hourly':
      threshold = STALENESS_THRESHOLDS.hourly;
      break;
    case 'daily':
      threshold = STALENESS_THRESHOLDS.daily;
      break;
    default:
      threshold = STALENESS_THRESHOLDS.daily;
  }

  if (age < threshold) return 'fresh';
  if (age < threshold * 2) return 'stale';
  return 'error';
}

/**
 * Get enabled financial sources
 */
export function getEnabledFinancialSources(): FinancialSource[] {
  return ALL_FINANCIAL_SOURCES.filter(s => s.enabled);
}

/**
 * Get sources by data type
 */
export function getSourcesByDataType(dataType: DataType): FinancialSource[] {
  return ALL_FINANCIAL_SOURCES.filter(
    s => s.enabled && s.dataTypes.includes(dataType)
  );
}

/**
 * Calculate total monthly API cost
 */
export function calculateFinancialAPICost(): number {
  return getEnabledFinancialSources().reduce((sum, s) => sum + s.costPerMonth, 0);
}
