import { EntropySource } from '../types';

// Free APIs: CoinGecko (Crypto) and OpenMeteo (Weather)
// No API Keys required for basic tiers.

export const fetchEntropy = async (): Promise<EntropySource[]> => {
  const sources: EntropySource[] = [];

  try {
    // 1. Economic Entropy (Crypto Volatility)
    const cryptoRes = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true');
    const cryptoData = await cryptoRes.json();
    
    if (cryptoData.bitcoin) {
      sources.push({
        id: 'BTC_VOL',
        source: 'CoinGecko',
        value: Math.abs(cryptoData.bitcoin.usd_24h_change) / 10, // Normalize rough 0-1
        label: 'BTC Flux',
        delta: cryptoData.bitcoin.usd_24h_change
      });
    }
  } catch (e) {
    console.warn("Crypto Entropy Offline, simulating...");
    sources.push({ id: 'BTC_SIM', source: 'System', value: Math.random(), label: 'Sim Flux', delta: (Math.random() - 0.5) * 5 });
  }

  try {
    // 2. Environmental Entropy (Atmospheric Noise)
    // Using Coordinates for a chaotic weather location (e.g., Tokyo)
    const weatherRes = await fetch('https://api.open-meteo.com/v1/forecast?latitude=35.6762&longitude=139.6503&current=temperature_2m,wind_speed_10m,pressure_msl');
    const weatherData = await weatherRes.json();

    if (weatherData.current) {
      sources.push({
        id: 'ATM_PRESS',
        source: 'OpenMeteo',
        value: (weatherData.current.pressure_msl - 980) / 50, // Normalize
        label: 'Atm. Pressure',
        delta: weatherData.current.wind_speed_10m
      });
    }
  } catch (e) {
    console.warn("Weather Entropy Offline, simulating...");
    sources.push({ id: 'ATM_SIM', source: 'System', value: Math.random(), label: 'Sim Atm', delta: (Math.random() - 0.5) * 2 });
  }

  return sources;
};