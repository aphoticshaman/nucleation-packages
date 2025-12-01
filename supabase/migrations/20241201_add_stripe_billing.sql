-- Add Stripe billing columns to clients table
ALTER TABLE clients
ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT UNIQUE,
ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT,
ADD COLUMN IF NOT EXISTS subscription_status TEXT DEFAULT 'none';

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_clients_stripe_customer ON clients(stripe_customer_id);

-- Update tier_limits with correct values
DELETE FROM tier_limits;

INSERT INTO tier_limits (tier, max_signal_tokens, max_fusion_tokens, max_analysis_tokens, max_total_tokens, max_api_keys, max_webhooks)
VALUES
  ('free',       5000,    5000,    5000,     15000,   1,   0),
  ('pro',       50000,   50000,   50000,    150000,   5,   3),
  ('enterprise', 500000, 500000,  500000,  1500000,  50,  25),
  ('government', 2147483647, 2147483647, 2147483647, 2147483647, 1000, 500);
