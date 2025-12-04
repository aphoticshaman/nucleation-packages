-- Nation Changes System
-- Handles nation splits, mergers, renames, and takeovers
-- Maintains historical data integrity for intelligence tracking

-- Add status and relationship fields to nations table
ALTER TABLE nations
ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'disputed')),
ADD COLUMN IF NOT EXISTS successor_codes TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS predecessor_codes TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS archived_reason TEXT;

-- Create nation_changes table to track all historical changes
CREATE TABLE IF NOT EXISTS nation_changes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  change_type TEXT NOT NULL CHECK (change_type IN ('split', 'merge', 'rename', 'takeover', 'independence', 'dissolution')),

  -- Source nation(s) - the nation(s) that changed
  source_codes TEXT[] NOT NULL,

  -- Result nation(s) - the nation(s) that resulted from the change
  result_codes TEXT[] NOT NULL,

  -- Change details
  effective_date DATE NOT NULL,
  description TEXT NOT NULL,

  -- For renames, track old and new names
  old_name TEXT,
  new_name TEXT,

  -- For territorial changes, track what was transferred
  territory_transferred TEXT,

  -- Data migration settings
  transfer_historical_data BOOLEAN DEFAULT true,
  data_weight_distribution JSONB, -- How to split historical data between successors

  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  created_by UUID REFERENCES auth.users(id),
  verified BOOLEAN DEFAULT false,
  verified_at TIMESTAMPTZ,
  verified_by UUID REFERENCES auth.users(id),

  -- Source documentation
  source_references TEXT[],
  notes TEXT
);

-- Create index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_nation_changes_source ON nation_changes USING GIN (source_codes);
CREATE INDEX IF NOT EXISTS idx_nation_changes_result ON nation_changes USING GIN (result_codes);
CREATE INDEX IF NOT EXISTS idx_nation_changes_date ON nation_changes (effective_date);
CREATE INDEX IF NOT EXISTS idx_nation_changes_type ON nation_changes (change_type);

-- Function to handle nation rename
CREATE OR REPLACE FUNCTION process_nation_rename(
  p_old_code TEXT,
  p_new_code TEXT,
  p_new_name TEXT,
  p_effective_date DATE DEFAULT CURRENT_DATE,
  p_description TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
  v_change_id UUID;
  v_old_nation RECORD;
BEGIN
  -- Get the old nation data
  SELECT * INTO v_old_nation FROM nations WHERE code = p_old_code;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Source nation % not found', p_old_code;
  END IF;

  -- If codes are different, create new nation and archive old
  IF p_old_code != p_new_code THEN
    -- Create new nation with old data
    INSERT INTO nations (
      code, name, region, basin_strength, transition_risk,
      position, velocity, last_event, status, predecessor_codes
    ) VALUES (
      p_new_code, p_new_name, v_old_nation.region, v_old_nation.basin_strength,
      v_old_nation.transition_risk, v_old_nation.position, v_old_nation.velocity,
      v_old_nation.last_event, 'active', ARRAY[p_old_code]
    ) ON CONFLICT (code) DO UPDATE SET
      name = EXCLUDED.name,
      predecessor_codes = EXCLUDED.predecessor_codes;

    -- Archive old nation
    UPDATE nations SET
      status = 'archived',
      archived_at = NOW(),
      archived_reason = 'Renamed to ' || p_new_name,
      successor_codes = ARRAY[p_new_code]
    WHERE code = p_old_code;

    -- Update all references in learning_events
    UPDATE learning_events SET nation_code = p_new_code WHERE nation_code = p_old_code;

    -- Update training_examples domain references
    UPDATE training_examples SET
      metadata = jsonb_set(metadata, '{nation_code}', to_jsonb(p_new_code))
    WHERE metadata->>'nation_code' = p_old_code;
  ELSE
    -- Just update the name
    UPDATE nations SET name = p_new_name WHERE code = p_old_code;
  END IF;

  -- Record the change
  INSERT INTO nation_changes (
    change_type, source_codes, result_codes, effective_date,
    description, old_name, new_name
  ) VALUES (
    'rename', ARRAY[p_old_code], ARRAY[p_new_code], p_effective_date,
    COALESCE(p_description, v_old_nation.name || ' renamed to ' || p_new_name),
    v_old_nation.name, p_new_name
  ) RETURNING id INTO v_change_id;

  RETURN v_change_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to handle nation split
CREATE OR REPLACE FUNCTION process_nation_split(
  p_source_code TEXT,
  p_result_nations JSONB, -- Array of {code, name, region, data_weight}
  p_effective_date DATE DEFAULT CURRENT_DATE,
  p_description TEXT DEFAULT NULL,
  p_keep_source BOOLEAN DEFAULT false -- Whether source nation continues to exist (e.g., Russia after USSR split)
) RETURNS UUID AS $$
DECLARE
  v_change_id UUID;
  v_source_nation RECORD;
  v_result_codes TEXT[] := '{}';
  v_nation JSONB;
  v_weight_distribution JSONB := '{}';
BEGIN
  -- Get source nation
  SELECT * INTO v_source_nation FROM nations WHERE code = p_source_code;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Source nation % not found', p_source_code;
  END IF;

  -- Create each result nation
  FOR v_nation IN SELECT * FROM jsonb_array_elements(p_result_nations)
  LOOP
    v_result_codes := array_append(v_result_codes, v_nation->>'code');
    v_weight_distribution := v_weight_distribution || jsonb_build_object(
      v_nation->>'code', COALESCE((v_nation->>'data_weight')::NUMERIC, 1.0 / jsonb_array_length(p_result_nations))
    );

    -- Insert new nation (inherit some properties from source)
    INSERT INTO nations (
      code, name, region, basin_strength, transition_risk,
      position, velocity, status, predecessor_codes
    ) VALUES (
      v_nation->>'code',
      v_nation->>'name',
      COALESCE(v_nation->>'region', v_source_nation.region),
      COALESCE((v_nation->>'basin_strength')::NUMERIC, v_source_nation.basin_strength * 0.7),
      COALESCE((v_nation->>'transition_risk')::NUMERIC, 0.6), -- Higher risk during transition
      v_source_nation.position, -- Start at same position
      '[0,0,0]'::JSONB, -- Reset velocity
      'active',
      ARRAY[p_source_code]
    ) ON CONFLICT (code) DO UPDATE SET
      predecessor_codes = array_append(
        COALESCE(nations.predecessor_codes, '{}'),
        p_source_code
      );
  END LOOP;

  -- Archive or update source nation
  IF p_keep_source THEN
    UPDATE nations SET
      successor_codes = v_result_codes,
      transition_risk = LEAST(v_source_nation.transition_risk + 0.2, 1.0)
    WHERE code = p_source_code;
  ELSE
    UPDATE nations SET
      status = 'archived',
      archived_at = NOW(),
      archived_reason = 'Split into: ' || array_to_string(v_result_codes, ', '),
      successor_codes = v_result_codes
    WHERE code = p_source_code;
  END IF;

  -- Record the change
  INSERT INTO nation_changes (
    change_type, source_codes, result_codes, effective_date,
    description, data_weight_distribution, transfer_historical_data
  ) VALUES (
    'split', ARRAY[p_source_code], v_result_codes, p_effective_date,
    COALESCE(p_description, v_source_nation.name || ' split into ' || array_to_string(v_result_codes, ', ')),
    v_weight_distribution, true
  ) RETURNING id INTO v_change_id;

  RETURN v_change_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to handle nation merger
CREATE OR REPLACE FUNCTION process_nation_merge(
  p_source_codes TEXT[],
  p_result_code TEXT,
  p_result_name TEXT,
  p_result_region TEXT DEFAULT NULL,
  p_effective_date DATE DEFAULT CURRENT_DATE,
  p_description TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
  v_change_id UUID;
  v_source_code TEXT;
  v_avg_basin NUMERIC := 0;
  v_avg_risk NUMERIC := 0;
  v_count INT := 0;
  v_region TEXT;
BEGIN
  -- Calculate averages from source nations
  FOR v_source_code IN SELECT unnest(p_source_codes)
  LOOP
    SELECT
      basin_strength, transition_risk, region
    INTO
      v_avg_basin, v_avg_risk, v_region
    FROM nations WHERE code = v_source_code;

    IF FOUND THEN
      v_avg_basin := v_avg_basin + COALESCE(v_avg_basin, 0.5);
      v_avg_risk := v_avg_risk + COALESCE(v_avg_risk, 0.5);
      v_count := v_count + 1;
    END IF;
  END LOOP;

  IF v_count > 0 THEN
    v_avg_basin := v_avg_basin / v_count;
    v_avg_risk := v_avg_risk / v_count;
  ELSE
    v_avg_basin := 0.5;
    v_avg_risk := 0.5;
  END IF;

  -- Create merged nation
  INSERT INTO nations (
    code, name, region, basin_strength, transition_risk,
    position, velocity, status, predecessor_codes
  ) VALUES (
    p_result_code,
    p_result_name,
    COALESCE(p_result_region, v_region, 'Unknown'),
    v_avg_basin,
    LEAST(v_avg_risk + 0.1, 1.0), -- Slightly elevated risk during merger
    '[0,0,0]'::JSONB,
    '[0,0,0]'::JSONB,
    'active',
    p_source_codes
  ) ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    predecessor_codes = EXCLUDED.predecessor_codes;

  -- Archive source nations
  UPDATE nations SET
    status = 'archived',
    archived_at = NOW(),
    archived_reason = 'Merged into ' || p_result_name,
    successor_codes = ARRAY[p_result_code]
  WHERE code = ANY(p_source_codes);

  -- Migrate learning events to new nation
  UPDATE learning_events SET nation_code = p_result_code
  WHERE nation_code = ANY(p_source_codes);

  -- Record the change
  INSERT INTO nation_changes (
    change_type, source_codes, result_codes, effective_date,
    description, transfer_historical_data
  ) VALUES (
    'merge', p_source_codes, ARRAY[p_result_code], p_effective_date,
    COALESCE(p_description, array_to_string(p_source_codes, ', ') || ' merged into ' || p_result_name),
    true
  ) RETURNING id INTO v_change_id;

  RETURN v_change_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to handle takeover/annexation
CREATE OR REPLACE FUNCTION process_nation_takeover(
  p_taken_code TEXT,
  p_taker_code TEXT,
  p_territory_name TEXT DEFAULT NULL,
  p_effective_date DATE DEFAULT CURRENT_DATE,
  p_description TEXT DEFAULT NULL,
  p_disputed BOOLEAN DEFAULT true -- Most takeovers are disputed internationally
) RETURNS UUID AS $$
DECLARE
  v_change_id UUID;
  v_taken_nation RECORD;
  v_taker_nation RECORD;
BEGIN
  -- Get both nations
  SELECT * INTO v_taken_nation FROM nations WHERE code = p_taken_code;
  SELECT * INTO v_taker_nation FROM nations WHERE code = p_taker_code;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Nation not found';
  END IF;

  -- Update taken nation status
  UPDATE nations SET
    status = CASE WHEN p_disputed THEN 'disputed' ELSE 'archived' END,
    archived_at = CASE WHEN NOT p_disputed THEN NOW() ELSE NULL END,
    archived_reason = 'Taken over by ' || v_taker_nation.name,
    successor_codes = ARRAY[p_taker_code],
    transition_risk = 0.95 -- Very high risk during takeover
  WHERE code = p_taken_code;

  -- Update taker nation
  UPDATE nations SET
    predecessor_codes = array_append(COALESCE(predecessor_codes, '{}'), p_taken_code),
    transition_risk = LEAST(transition_risk + 0.3, 1.0) -- Elevated risk for aggressor too
  WHERE code = p_taker_code;

  -- Record the change
  INSERT INTO nation_changes (
    change_type, source_codes, result_codes, effective_date,
    description, territory_transferred
  ) VALUES (
    'takeover', ARRAY[p_taken_code], ARRAY[p_taker_code], p_effective_date,
    COALESCE(p_description, v_taken_nation.name || ' taken over by ' || v_taker_nation.name),
    COALESCE(p_territory_name, v_taken_nation.name)
  ) RETURNING id INTO v_change_id;

  RETURN v_change_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get nation history including predecessors/successors
CREATE OR REPLACE FUNCTION get_nation_history(p_code TEXT)
RETURNS TABLE (
  code TEXT,
  name TEXT,
  status TEXT,
  relationship TEXT,
  change_date DATE,
  change_type TEXT
) AS $$
BEGIN
  -- Current nation
  RETURN QUERY
  SELECT n.code, n.name, n.status, 'current'::TEXT, NULL::DATE, NULL::TEXT
  FROM nations n WHERE n.code = p_code;

  -- Predecessors
  RETURN QUERY
  SELECT n.code, n.name, n.status, 'predecessor'::TEXT, nc.effective_date, nc.change_type
  FROM nations n
  JOIN nation_changes nc ON n.code = ANY(nc.source_codes)
  WHERE p_code = ANY(nc.result_codes);

  -- Successors
  RETURN QUERY
  SELECT n.code, n.name, n.status, 'successor'::TEXT, nc.effective_date, nc.change_type
  FROM nations n
  JOIN nation_changes nc ON n.code = ANY(nc.result_codes)
  WHERE p_code = ANY(nc.source_codes);
END;
$$ LANGUAGE plpgsql;

-- View for active nations only (what the map should show)
CREATE OR REPLACE VIEW active_nations AS
SELECT * FROM nations WHERE status = 'active';

-- View for disputed territories (show differently on map)
CREATE OR REPLACE VIEW disputed_nations AS
SELECT * FROM nations WHERE status = 'disputed';

-- Grant permissions
GRANT SELECT ON nation_changes TO authenticated;
GRANT SELECT ON active_nations TO authenticated;
GRANT SELECT ON disputed_nations TO authenticated;
GRANT EXECUTE ON FUNCTION get_nation_history TO authenticated;

-- Admin-only functions
GRANT EXECUTE ON FUNCTION process_nation_rename TO service_role;
GRANT EXECUTE ON FUNCTION process_nation_split TO service_role;
GRANT EXECUTE ON FUNCTION process_nation_merge TO service_role;
GRANT EXECUTE ON FUNCTION process_nation_takeover TO service_role;

-- Seed some historical examples (can be used for testing)
-- Note: These are historical examples, not political statements

COMMENT ON TABLE nation_changes IS 'Tracks historical changes to nations including splits, mergers, renames, and takeovers. Used to maintain data integrity when nation boundaries change.';
COMMENT ON FUNCTION process_nation_rename IS 'Handles nation rename (e.g., Swaziland → Eswatini). Updates all references and maintains historical link.';
COMMENT ON FUNCTION process_nation_split IS 'Handles nation split (e.g., USSR → Russia, Ukraine, etc.). Creates new nations and optionally archives source.';
COMMENT ON FUNCTION process_nation_merge IS 'Handles nation merger (e.g., East + West Germany → Germany). Combines data and archives sources.';
COMMENT ON FUNCTION process_nation_takeover IS 'Handles annexation/takeover. Marks territory as disputed and links to new controller.';
