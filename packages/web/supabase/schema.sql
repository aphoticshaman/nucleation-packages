-- LatticeForge Supabase Schema
-- Nation-level attractor dynamics with Google Maps integration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- ============================================
-- CORE TABLES
-- ============================================

-- Nations with geographic and attractor state
CREATE TABLE nations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(3) UNIQUE NOT NULL,           -- ISO 3166-1 alpha-3
    name VARCHAR(255) NOT NULL,

    -- Geographic coordinates
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    geometry GEOMETRY(Point, 4326),            -- PostGIS point

    -- Attractor state (4D position vector stored as array)
    position DOUBLE PRECISION[] NOT NULL DEFAULT '{0.5, 0.5, 0.5, 0.5}',
    velocity DOUBLE PRECISION[] NOT NULL DEFAULT '{0, 0, 0, 0}',

    -- Computed metrics
    basin_strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    transition_risk DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    regime INTEGER NOT NULL DEFAULT 0,
    influence_radius DOUBLE PRECISION NOT NULL DEFAULT 1.0,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trigger to auto-update geometry from lat/lon
CREATE OR REPLACE FUNCTION update_nation_geometry()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geometry := ST_SetSRID(ST_MakePoint(NEW.lon, NEW.lat), 4326);
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER nation_geometry_trigger
    BEFORE INSERT OR UPDATE OF lat, lon ON nations
    FOR EACH ROW EXECUTE FUNCTION update_nation_geometry();

-- Esteem matrix: how each nation views others
CREATE TABLE esteem_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES nations(id) ON DELETE CASCADE,
    target_id UUID REFERENCES nations(id) ON DELETE CASCADE,
    esteem DOUBLE PRECISION NOT NULL CHECK (esteem >= -1 AND esteem <= 1),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_id, target_id)
);

-- Influence edges (computed from simulation)
CREATE TABLE influence_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES nations(id) ON DELETE CASCADE,
    target_id UUID REFERENCES nations(id) ON DELETE CASCADE,
    strength DOUBLE PRECISION NOT NULL,
    geodesic_distance DOUBLE PRECISION NOT NULL,
    esteem DOUBLE PRECISION,
    simulation_id UUID,                        -- Links to simulation run
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(source_id, target_id, simulation_id)
);

-- ============================================
-- SIMULATION TABLES
-- ============================================

-- Simulation configurations
CREATE TABLE simulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),

    -- Config
    n_dims INTEGER NOT NULL DEFAULT 4,
    interaction_decay DOUBLE PRECISION NOT NULL DEFAULT 0.001,
    min_influence DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    dt DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    diffusion DOUBLE PRECISION NOT NULL DEFAULT 0.05,

    -- State
    sim_time DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    n_steps INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'created',

    -- Metadata
    user_id UUID,                              -- For multi-user support
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Simulation snapshots (time-series state)
CREATE TABLE simulation_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    simulation_id UUID REFERENCES simulations(id) ON DELETE CASCADE,
    time_step DOUBLE PRECISION NOT NULL,

    -- Full state as JSONB for flexibility
    nations_state JSONB NOT NULL,              -- {code: {position, velocity, basin_strength, ...}}
    n_edges INTEGER NOT NULL DEFAULT 0,

    -- TDA metrics
    persistent_entropy DOUBLE PRECISION,
    h0_count INTEGER,
    h1_count INTEGER,

    -- Alert level
    alert_level VARCHAR(50),
    phase_transition_probability DOUBLE PRECISION,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for time-series queries
CREATE INDEX idx_snapshots_simulation_time
    ON simulation_snapshots(simulation_id, time_step);

-- ============================================
-- REGIME TABLES
-- ============================================

-- Regime definitions
CREATE TABLE regimes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    color VARCHAR(7),                          -- Hex color for visualization
    metadata JSONB DEFAULT '{}'
);

-- Pre-populate some regime types
INSERT INTO regimes (id, name, description, color) VALUES
    (0, 'Liberal Democracy', 'Open markets, individual rights', '#3B82F6'),
    (1, 'State Capitalism', 'Strong state economic control', '#EF4444'),
    (2, 'Social Democracy', 'Mixed economy with welfare', '#10B981'),
    (3, 'Authoritarian', 'Centralized political control', '#6B7280'),
    (4, 'Transitional', 'Unstable/changing regime', '#F59E0B');

-- ============================================
-- GEOSPATIAL VIEWS
-- ============================================

-- GeoJSON export view for Google Maps
CREATE OR REPLACE VIEW nations_geojson AS
SELECT jsonb_build_object(
    'type', 'FeatureCollection',
    'features', jsonb_agg(
        jsonb_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON(geometry)::jsonb,
            'properties', jsonb_build_object(
                'code', code,
                'name', name,
                'basin_strength', basin_strength,
                'transition_risk', transition_risk,
                'regime', regime,
                'position', position
            )
        )
    )
) AS geojson
FROM nations;

-- Influence edges as GeoJSON lines
CREATE OR REPLACE VIEW edges_geojson AS
SELECT jsonb_build_object(
    'type', 'FeatureCollection',
    'features', jsonb_agg(
        jsonb_build_object(
            'type', 'Feature',
            'geometry', jsonb_build_object(
                'type', 'LineString',
                'coordinates', jsonb_build_array(
                    jsonb_build_array(s.lon, s.lat),
                    jsonb_build_array(t.lon, t.lat)
                )
            ),
            'properties', jsonb_build_object(
                'source', s.code,
                'target', t.code,
                'strength', e.strength,
                'esteem', e.esteem
            )
        )
    )
) AS geojson
FROM influence_edges e
JOIN nations s ON e.source_id = s.id
JOIN nations t ON e.target_id = t.id
WHERE e.strength > 0.1;

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================

-- Enable RLS
ALTER TABLE nations ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_snapshots ENABLE ROW LEVEL SECURITY;

-- Public read for nations
CREATE POLICY "Nations are viewable by everyone"
    ON nations FOR SELECT USING (true);

-- Simulations belong to users
CREATE POLICY "Users can view their own simulations"
    ON simulations FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can create simulations"
    ON simulations FOR INSERT
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update their own simulations"
    ON simulations FOR UPDATE
    USING (auth.uid() = user_id);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Get nations within distance (km) of a point
CREATE OR REPLACE FUNCTION nations_within_distance(
    center_lat DOUBLE PRECISION,
    center_lon DOUBLE PRECISION,
    distance_km DOUBLE PRECISION
)
RETURNS SETOF nations AS $$
BEGIN
    RETURN QUERY
    SELECT *
    FROM nations
    WHERE ST_DWithin(
        geometry::geography,
        ST_SetSRID(ST_MakePoint(center_lon, center_lat), 4326)::geography,
        distance_km * 1000
    );
END;
$$ LANGUAGE plpgsql;

-- Get nation comparison
CREATE OR REPLACE FUNCTION compare_nations(code1 VARCHAR, code2 VARCHAR)
RETURNS JSONB AS $$
DECLARE
    n1 nations;
    n2 nations;
    geo_dist DOUBLE PRECISION;
    attr_dist DOUBLE PRECISION;
    esteem_12 DOUBLE PRECISION;
    esteem_21 DOUBLE PRECISION;
BEGIN
    SELECT * INTO n1 FROM nations WHERE code = code1;
    SELECT * INTO n2 FROM nations WHERE code = code2;

    IF n1 IS NULL OR n2 IS NULL THEN
        RETURN jsonb_build_object('error', 'Nation not found');
    END IF;

    -- Geographic distance in km
    geo_dist := ST_Distance(n1.geometry::geography, n2.geometry::geography) / 1000;

    -- Attractor space distance (Euclidean)
    SELECT SQRT(SUM(POWER(a - b, 2)))
    INTO attr_dist
    FROM UNNEST(n1.position, n2.position) AS t(a, b);

    -- Get esteem values
    SELECT esteem INTO esteem_12 FROM esteem_relations
        WHERE source_id = n1.id AND target_id = n2.id;
    SELECT esteem INTO esteem_21 FROM esteem_relations
        WHERE source_id = n2.id AND target_id = n1.id;

    RETURN jsonb_build_object(
        'nations', jsonb_build_array(code1, code2),
        'geographic_distance_km', geo_dist,
        'attractor_distance', attr_dist,
        'esteem', jsonb_build_object(
            code1 || '_views_' || code2, COALESCE(esteem_12, 0),
            code2 || '_views_' || code1, COALESCE(esteem_21, 0)
        ),
        'regime_aligned', n1.regime = n2.regime
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- SEED DATA (Major nations)
-- ============================================

INSERT INTO nations (code, name, lat, lon, regime, position) VALUES
    ('USA', 'United States', 39.8283, -98.5795, 0, '{0.8, 0.3, 0.7, 0.6}'),
    ('CHN', 'China', 35.8617, 104.1954, 1, '{0.3, 0.9, 0.4, 0.7}'),
    ('RUS', 'Russia', 61.5240, 105.3188, 3, '{0.4, 0.7, 0.3, 0.5}'),
    ('GBR', 'United Kingdom', 55.3781, -3.4360, 0, '{0.75, 0.35, 0.65, 0.55}'),
    ('DEU', 'Germany', 51.1657, 10.4515, 2, '{0.7, 0.4, 0.6, 0.6}'),
    ('FRA', 'France', 46.2276, 2.2137, 2, '{0.7, 0.45, 0.55, 0.6}'),
    ('JPN', 'Japan', 36.2048, 138.2529, 0, '{0.75, 0.5, 0.7, 0.65}'),
    ('IND', 'India', 20.5937, 78.9629, 0, '{0.6, 0.5, 0.5, 0.4}'),
    ('BRA', 'Brazil', -14.2350, -51.9253, 0, '{0.55, 0.4, 0.45, 0.35}'),
    ('AUS', 'Australia', -25.2744, 133.7751, 0, '{0.78, 0.32, 0.68, 0.58}'),
    ('KOR', 'South Korea', 35.9078, 127.7669, 0, '{0.72, 0.45, 0.65, 0.6}'),
    ('SAU', 'Saudi Arabia', 23.8859, 45.0792, 3, '{0.35, 0.6, 0.3, 0.4}'),
    ('IRN', 'Iran', 32.4279, 53.6880, 3, '{0.25, 0.7, 0.2, 0.35}'),
    ('ISR', 'Israel', 31.0461, 34.8516, 0, '{0.7, 0.5, 0.6, 0.55}'),
    ('TUR', 'Turkey', 38.9637, 35.2433, 4, '{0.5, 0.55, 0.45, 0.5}'),
    ('MEX', 'Mexico', 23.6345, -102.5528, 0, '{0.5, 0.4, 0.45, 0.4}'),
    ('IDN', 'Indonesia', -0.7893, 113.9213, 0, '{0.55, 0.45, 0.5, 0.45}'),
    ('NGA', 'Nigeria', 9.0820, 8.6753, 0, '{0.45, 0.4, 0.35, 0.3}'),
    ('ZAF', 'South Africa', -30.5595, 22.9375, 0, '{0.5, 0.35, 0.4, 0.35}'),
    ('EGY', 'Egypt', 26.8206, 30.8025, 3, '{0.4, 0.55, 0.35, 0.4}');

-- Seed esteem relations (major relationships)
INSERT INTO esteem_relations (source_id, target_id, esteem)
SELECT s.id, t.id, e.esteem
FROM (VALUES
    ('USA', 'GBR', 0.85),
    ('GBR', 'USA', 0.80),
    ('USA', 'CHN', -0.35),
    ('CHN', 'USA', -0.30),
    ('USA', 'RUS', -0.50),
    ('RUS', 'USA', -0.55),
    ('USA', 'JPN', 0.70),
    ('JPN', 'USA', 0.65),
    ('DEU', 'FRA', 0.75),
    ('FRA', 'DEU', 0.70),
    ('CHN', 'RUS', 0.60),
    ('RUS', 'CHN', 0.55),
    ('SAU', 'IRN', -0.80),
    ('IRN', 'SAU', -0.85),
    ('ISR', 'USA', 0.75),
    ('USA', 'ISR', 0.70)
) AS e(source_code, target_code, esteem)
JOIN nations s ON s.code = e.source_code
JOIN nations t ON t.code = e.target_code;
