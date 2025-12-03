-- Integrations table for Slack, Teams, webhooks, etc.
-- Migration: 20241203000003_integrations

-- 1. Create integrations table
CREATE TABLE IF NOT EXISTS integrations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,

  -- Integration type and provider
  type TEXT NOT NULL CHECK (type IN ('slack', 'teams', 'webhook', 'email')),

  -- Connection status
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'error', 'disconnected')),

  -- OAuth tokens (encrypted in production)
  access_token TEXT,
  refresh_token TEXT,
  token_expires_at TIMESTAMPTZ,

  -- Provider-specific data
  -- Slack: { team_id, team_name, channel_id, channel_name, webhook_url, bot_user_id }
  -- Teams: { tenant_id, team_id, channel_id, webhook_url }
  -- Webhook: { url, secret, headers }
  provider_data JSONB NOT NULL DEFAULT '{}',

  -- Alert configuration
  -- { categories: [...], severity_threshold: 'elevated', daily_digest: true, ... }
  alert_config JSONB NOT NULL DEFAULT '{
    "enabled": true,
    "categories": ["security", "political", "economic"],
    "severity_threshold": "elevated",
    "daily_digest": false,
    "breaking_news": true
  }',

  -- Metadata
  created_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_used_at TIMESTAMPTZ,
  error_message TEXT,

  -- Ensure one integration per type per org
  CONSTRAINT integrations_org_type_unique UNIQUE (organization_id, type)
);

-- 2. Create integration_logs table for audit trail
CREATE TABLE IF NOT EXISTS integration_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  integration_id UUID NOT NULL REFERENCES integrations(id) ON DELETE CASCADE,

  -- Log entry
  event_type TEXT NOT NULL, -- 'message_sent', 'auth_refresh', 'error', 'config_change'
  event_data JSONB NOT NULL DEFAULT '{}',

  -- Result
  success BOOLEAN NOT NULL DEFAULT true,
  error_message TEXT,

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Enable RLS
ALTER TABLE integrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE integration_logs ENABLE ROW LEVEL SECURITY;

-- 4. RLS policies for integrations
-- Users can view integrations for their org
CREATE POLICY "Users can view org integrations"
  ON integrations FOR SELECT
  USING (
    organization_id IN (
      SELECT organization_id FROM profiles WHERE id = auth.uid()
    )
  );

-- Only admins/enterprise can manage integrations
CREATE POLICY "Admins can manage integrations"
  ON integrations FOR ALL
  USING (
    organization_id IN (
      SELECT organization_id FROM profiles
      WHERE id = auth.uid()
      AND (role = 'admin' OR role = 'enterprise')
    )
  );

-- 5. RLS policies for integration_logs
CREATE POLICY "Users can view org integration logs"
  ON integration_logs FOR SELECT
  USING (
    integration_id IN (
      SELECT i.id FROM integrations i
      JOIN profiles p ON p.organization_id = i.organization_id
      WHERE p.id = auth.uid()
    )
  );

-- 6. Indexes
CREATE INDEX IF NOT EXISTS idx_integrations_org ON integrations(organization_id);
CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(type);
CREATE INDEX IF NOT EXISTS idx_integrations_status ON integrations(status);
CREATE INDEX IF NOT EXISTS idx_integration_logs_integration ON integration_logs(integration_id);
CREATE INDEX IF NOT EXISTS idx_integration_logs_created ON integration_logs(created_at DESC);

-- 7. Update trigger
CREATE OR REPLACE FUNCTION update_integrations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS integrations_updated_at ON integrations;
CREATE TRIGGER integrations_updated_at
  BEFORE UPDATE ON integrations
  FOR EACH ROW
  EXECUTE FUNCTION update_integrations_updated_at();

-- 8. Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON integrations TO authenticated;
GRANT SELECT, INSERT ON integration_logs TO authenticated;

COMMENT ON TABLE integrations IS 'Third-party integrations (Slack, Teams, webhooks) per organization';
COMMENT ON TABLE integration_logs IS 'Audit log for integration events';
