# Vercel Backend Deployment Guide

## Overview

The VeilPath Tarot backend runs on **Vercel Serverless Functions** and uses:
- **Anthropic Claude API** for tarot interpretations
- **Neon Serverless Postgres** for caching and persistence
- **Node.js 20** runtime

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Anthropic API Key**: Get one at [console.anthropic.com](https://console.anthropic.com)
3. **Neon Database**: Create a free database at [neon.tech](https://neon.tech)

## Step 1: Set Up Neon Database

1. Create a new Neon project at [console.neon.tech](https://console.neon.tech)
2. Copy your connection string (format: `postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`)
3. Run the database schema:

```bash
# Install Neon CLI
npm install -g neonctl

# Or use psql directly
psql $DATABASE_URL -f db/schema.sql
```

4. Verify tables were created:
```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public';
```

You should see: `readings`, `llm_cache`, `user_profiles`, `currency_transactions`

## Step 2: Deploy to Vercel

### Option A: CLI Deployment (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

### Option B: GitHub Integration

1. Push your code to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your GitHub repository
4. Vercel will auto-detect the project configuration

## Step 3: Configure Environment Variables

In your Vercel project dashboard:

1. Go to **Settings** → **Environment Variables**
2. Add these variables:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx
DATABASE_URL=postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require
ENABLE_LLM=true
CACHE_TTL_SECONDS=604800
NODE_ENV=production
```

3. Click **Save**
4. **Redeploy** your project to apply the new environment variables

## Step 4: Update React Native App

1. Copy your Vercel deployment URL (e.g., `https://hungry-orca.vercel.app`)
2. Create a `.env` file in your project root:

```bash
EXPO_PUBLIC_API_URL=https://hungry-orca.vercel.app
```

3. Restart your Expo dev server:

```bash
npm start -- --clear
```

## Testing Your Deployment

### Test Endpoints

1. **Health check** (should return API info):
```bash
curl https://your-app.vercel.app/api
```

2. **Interpret a card**:
```bash
curl -X POST https://your-app.vercel.app/api/interpret-card \
  -H "Content-Type: application/json" \
  -d '{
    "card": {
      "id": 0,
      "name": "The Fool",
      "arcana": "major",
      "keywords": {
        "upright": ["new beginnings", "innocence", "adventure"]
      }
    },
    "context": {
      "intention": "What guidance do I need?",
      "spreadType": "single_card"
    }
  }'
```

3. **Luna narration**:
```bash
curl -X POST https://your-app.vercel.app/api/narrate/luna \
  -H "Content-Type: application/json" \
  -d '{
    "moment": "reading_start",
    "context": {
      "intention": "seeking clarity"
    }
  }'
```

## Monitoring & Optimization

### Check Cache Performance

```sql
SELECT * FROM cache_stats;
```

Expected cache hit rate: 80%+ after initial usage

### Monitor Token Usage

```sql
SELECT
  model,
  SUM(tokens_used) as total_tokens,
  COUNT(*) as request_count,
  AVG(tokens_used) as avg_tokens
FROM llm_cache
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY model;
```

### Cost Estimation

With 80% cache hit rate:
- **Single card interpretation**: ~$0.003 (uncached) / $0 (cached)
- **Three card reading**: ~$0.01 (uncached) / $0 (cached)
- **Expected monthly cost** (1000 readings): ~$2-5

### Cleanup Expired Cache

Set up a Vercel Cron Job to clean up expired cache daily:

1. Create `api/cron/cleanup-cache.js`:
```javascript
import { cleanupExpiredCache } from '../../lib/cache.js';

export default async function handler(req, res) {
  const count = await cleanupExpiredCache();
  res.json({ deleted: count });
}
```

2. Add to `vercel.json`:
```json
{
  "crons": [{
    "path": "/api/cron/cleanup-cache",
    "schedule": "0 0 * * *"
  }]
}
```

## Troubleshooting

### "LLM features are disabled"

- Check that `ANTHROPIC_API_KEY` is set in Vercel environment variables
- Verify `ENABLE_LLM=true` is set
- Redeploy after changing environment variables

### "Failed to connect to database"

- Verify `DATABASE_URL` is correct
- Check that Neon database is running
- Ensure IP whitelist allows Vercel (Neon serverless allows all by default)

### "Too many requests" (429 error)

- Rate limits are configured per endpoint
- Default: 10/min for interpretations, 5/min for synthesis
- Adjust in `api/_middleware.js` if needed

### Cache not working

- Check database connection
- Verify `llm_cache` table exists
- Check logs for cache errors in Vercel dashboard

## Local Development

Run Vercel functions locally:

```bash
# Install dependencies
npm install

# Start Vercel dev server
vercel dev
```

This will start a local server at `http://localhost:3000`

Update your `.env`:
```bash
EXPO_PUBLIC_API_URL=http://localhost:3000
```

## Next Steps

- [ ] Set up custom domain (optional)
- [ ] Configure Vercel Analytics
- [ ] Set up error monitoring (Sentry, LogRocket)
- [ ] Implement request authentication (Phase 4)
- [ ] Add user-specific caching (Phase 4)

## Resources

- [Vercel Serverless Functions Docs](https://vercel.com/docs/functions)
- [Anthropic Claude API Docs](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Neon Serverless Postgres Docs](https://neon.tech/docs/introduction)
