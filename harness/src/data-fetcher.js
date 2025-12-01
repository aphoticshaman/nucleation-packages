/**
 * Data Fetcher - Real-time data from free public APIs
 * 
 * Sources:
 * - CoinGecko: Crypto prices (free, no auth)
 * - FRED: Federal Reserve economic data (free, optional API key)
 * - USGS: Earthquake data (free, no auth)
 * - Reddit: Public subreddit sentiment (free, no auth for public)
 * - GitHub: Repository activity (free, limited)
 * - NewsAPI: Headlines (free tier)
 */

const DATA_SOURCES = {
  // Crypto - CoinGecko (free, no auth, 10-30 calls/min)
  crypto: {
    bitcoin: 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=90',
    ethereum: 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=90',
    global: 'https://api.coingecko.com/api/v3/global'
  },
  
  // Fear & Greed Index (free, no auth)
  sentiment: {
    cryptoFear: 'https://api.alternative.me/fng/?limit=90'
  },
  
  // USGS Earthquakes (free, no auth) - proxy for geophysical instability
  geophysical: {
    earthquakes: 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson'
  },
  
  // GitHub API (free, 60 req/hour unauthenticated)
  tech: {
    trending: 'https://api.github.com/search/repositories?q=created:>2024-01-01&sort=stars&per_page=10'
  }
};

/**
 * Fetch with retry and error handling
 */
async function fetchWithRetry(url, retries = 3, delay = 1000) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'nucleation-harness/1.0' }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(r => setTimeout(r, delay * (i + 1)));
    }
  }
}

/**
 * Fetch crypto market data
 */
export async function fetchCryptoData() {
  const results = {};
  
  try {
    // Bitcoin price history
    const btc = await fetchWithRetry(DATA_SOURCES.crypto.bitcoin);
    results.bitcoin = {
      prices: btc.prices.map(([ts, price]) => ({ timestamp: ts, price })),
      volumes: btc.total_volumes.map(([ts, vol]) => ({ timestamp: ts, volume: vol }))
    };
    
    // Ethereum price history
    const eth = await fetchWithRetry(DATA_SOURCES.crypto.ethereum);
    results.ethereum = {
      prices: eth.prices.map(([ts, price]) => ({ timestamp: ts, price })),
      volumes: eth.total_volumes.map(([ts, vol]) => ({ timestamp: ts, volume: vol }))
    };
    
    // Global market data
    const global = await fetchWithRetry(DATA_SOURCES.crypto.global);
    results.global = {
      totalMarketCap: global.data.total_market_cap.usd,
      marketCapChange24h: global.data.market_cap_change_percentage_24h_usd,
      btcDominance: global.data.market_cap_percentage.btc
    };
    
  } catch (error) {
    results.error = error.message;
  }
  
  return results;
}

/**
 * Fetch sentiment data (Fear & Greed)
 */
export async function fetchSentimentData() {
  try {
    const data = await fetchWithRetry(DATA_SOURCES.sentiment.cryptoFear);
    return {
      current: parseInt(data.data[0].value),
      classification: data.data[0].value_classification,
      history: data.data.map(d => ({
        timestamp: parseInt(d.timestamp) * 1000,
        value: parseInt(d.value),
        classification: d.value_classification
      }))
    };
  } catch (error) {
    return { error: error.message };
  }
}

/**
 * Fetch geophysical data (earthquakes as instability proxy)
 */
export async function fetchGeophysicalData() {
  try {
    const data = await fetchWithRetry(DATA_SOURCES.geophysical.earthquakes);
    
    const quakes = data.features.map(f => ({
      magnitude: f.properties.mag,
      place: f.properties.place,
      time: f.properties.time,
      depth: f.geometry.coordinates[2]
    }));
    
    // Aggregate by day
    const byDay = {};
    quakes.forEach(q => {
      const day = new Date(q.time).toISOString().split('T')[0];
      if (!byDay[day]) byDay[day] = { count: 0, totalMag: 0, maxMag: 0 };
      byDay[day].count++;
      byDay[day].totalMag += q.magnitude || 0;
      byDay[day].maxMag = Math.max(byDay[day].maxMag, q.magnitude || 0);
    });
    
    return {
      total: quakes.length,
      significant: quakes.filter(q => q.magnitude >= 4.5).length,
      byDay,
      recentLarge: quakes.filter(q => q.magnitude >= 5.0).slice(0, 5)
    };
  } catch (error) {
    return { error: error.message };
  }
}

/**
 * Convert price data to returns for regime detection
 */
export function pricesToReturns(prices) {
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push({
      timestamp: prices[i].timestamp,
      value: Math.log(prices[i].price / prices[i - 1].price)
    });
  }
  return returns;
}

/**
 * Fetch all data sources
 */
export async function fetchAllData() {
  console.log('Fetching real-time data from public APIs...\n');
  
  const [crypto, sentiment, geophysical] = await Promise.all([
    fetchCryptoData(),
    fetchSentimentData(),
    fetchGeophysicalData()
  ]);
  
  return {
    fetchedAt: new Date().toISOString(),
    crypto,
    sentiment,
    geophysical
  };
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  fetchAllData()
    .then(data => {
      console.log(JSON.stringify(data, null, 2));
    })
    .catch(console.error);
}

export default fetchAllData;
