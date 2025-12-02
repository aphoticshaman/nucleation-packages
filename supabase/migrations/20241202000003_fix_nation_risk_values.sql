-- Fix nation risk values and add fragile states near phase transition
-- The original seed data left basin_strength=1.0 and transition_risk=0.0 (all green)

-- ============================================
-- UPDATE EXISTING NATIONS WITH REALISTIC VALUES
-- ============================================

-- Compute basin_strength from position vector:
-- Lower values in position dimensions = lower stability
-- Higher variance in position = lower stability

UPDATE nations SET
  basin_strength = CASE code
    -- Stable democracies (0.7-0.9)
    WHEN 'USA' THEN 0.78
    WHEN 'GBR' THEN 0.82
    WHEN 'DEU' THEN 0.85
    WHEN 'FRA' THEN 0.80
    WHEN 'JPN' THEN 0.88
    WHEN 'AUS' THEN 0.86
    WHEN 'KOR' THEN 0.75

    -- Emerging/mixed (0.4-0.7)
    WHEN 'IND' THEN 0.58
    WHEN 'BRA' THEN 0.52
    WHEN 'MEX' THEN 0.48
    WHEN 'IDN' THEN 0.55
    WHEN 'ZAF' THEN 0.45
    WHEN 'NGA' THEN 0.38

    -- Authoritarian stable (0.5-0.7)
    WHEN 'CHN' THEN 0.72
    WHEN 'RUS' THEN 0.58
    WHEN 'SAU' THEN 0.65

    -- Authoritarian stressed (0.3-0.5)
    WHEN 'IRN' THEN 0.42
    WHEN 'EGY' THEN 0.48

    -- Transitional (0.3-0.5)
    WHEN 'TUR' THEN 0.45
    WHEN 'ISR' THEN 0.62

    ELSE 0.5
  END,

  transition_risk = CASE code
    -- Low risk (0.05-0.2)
    WHEN 'USA' THEN 0.18
    WHEN 'GBR' THEN 0.12
    WHEN 'DEU' THEN 0.08
    WHEN 'FRA' THEN 0.15
    WHEN 'JPN' THEN 0.06
    WHEN 'AUS' THEN 0.07
    WHEN 'CHN' THEN 0.22

    -- Moderate risk (0.2-0.4)
    WHEN 'KOR' THEN 0.25
    WHEN 'IND' THEN 0.32
    WHEN 'BRA' THEN 0.38
    WHEN 'RUS' THEN 0.35
    WHEN 'SAU' THEN 0.28

    -- Elevated risk (0.4-0.6)
    WHEN 'MEX' THEN 0.42
    WHEN 'IDN' THEN 0.35
    WHEN 'ZAF' THEN 0.48
    WHEN 'TUR' THEN 0.52
    WHEN 'IRN' THEN 0.55
    WHEN 'EGY' THEN 0.45
    WHEN 'ISR' THEN 0.42

    -- High risk (0.5-0.7)
    WHEN 'NGA' THEN 0.58

    ELSE 0.3
  END,

  updated_at = NOW()
WHERE code IN ('USA', 'CHN', 'RUS', 'GBR', 'DEU', 'FRA', 'JPN', 'IND', 'BRA',
               'AUS', 'KOR', 'SAU', 'IRN', 'ISR', 'TUR', 'MEX', 'IDN', 'NGA', 'ZAF', 'EGY');

-- ============================================
-- ADD FRAGILE STATES NEAR PHASE TRANSITION
-- ============================================

INSERT INTO nations (code, name, lat, lon, regime, position, basin_strength, transition_risk, velocity) VALUES
  -- Active conflict / collapse (transition_risk > 0.8)
  ('SYR', 'Syria', 34.8021, 38.9968, 4, '{0.15, 0.85, 0.1, 0.2}', 0.12, 0.92, '{-0.02, 0.01, -0.01, 0.0}'),
  ('YEM', 'Yemen', 15.5527, 48.5164, 4, '{0.1, 0.9, 0.1, 0.15}', 0.08, 0.95, '{-0.03, 0.02, -0.02, -0.01}'),
  ('AFG', 'Afghanistan', 33.9391, 67.7100, 3, '{0.1, 0.95, 0.05, 0.1}', 0.15, 0.88, '{0.01, 0.0, 0.0, 0.0}'),
  ('SDN', 'Sudan', 12.8628, 30.2176, 4, '{0.12, 0.88, 0.08, 0.18}', 0.10, 0.91, '{-0.02, 0.01, -0.01, 0.0}'),
  ('SSD', 'South Sudan', 6.877, 31.307, 4, '{0.08, 0.92, 0.05, 0.12}', 0.06, 0.94, '{-0.01, 0.0, -0.01, 0.0}'),
  ('MMR', 'Myanmar', 21.9162, 95.9560, 3, '{0.18, 0.82, 0.12, 0.22}', 0.18, 0.85, '{-0.02, 0.01, -0.01, 0.0}'),
  ('HTI', 'Haiti', 18.9712, -72.2852, 4, '{0.12, 0.7, 0.15, 0.18}', 0.11, 0.89, '{-0.02, 0.0, -0.01, 0.0}'),
  ('LBY', 'Libya', 26.3351, 17.2283, 4, '{0.15, 0.8, 0.1, 0.2}', 0.14, 0.86, '{0.0, 0.0, 0.01, 0.0}'),
  ('SOM', 'Somalia', 5.1521, 46.1996, 4, '{0.08, 0.88, 0.08, 0.1}', 0.09, 0.90, '{0.01, -0.01, 0.0, 0.0}'),

  -- High stress / approaching transition (transition_risk 0.6-0.8)
  ('VEN', 'Venezuela', 6.4238, -66.5897, 3, '{0.2, 0.78, 0.18, 0.25}', 0.22, 0.78, '{-0.01, 0.01, -0.01, 0.0}'),
  ('LBN', 'Lebanon', 33.8547, 35.8623, 4, '{0.3, 0.65, 0.25, 0.35}', 0.25, 0.75, '{-0.02, 0.01, -0.01, 0.0}'),
  ('PAK', 'Pakistan', 30.3753, 69.3451, 0, '{0.35, 0.6, 0.3, 0.35}', 0.32, 0.68, '{0.0, 0.01, 0.0, 0.0}'),
  ('ETH', 'Ethiopia', 9.1450, 40.4897, 3, '{0.28, 0.7, 0.22, 0.3}', 0.28, 0.72, '{0.01, 0.0, 0.0, 0.0}'),
  ('UKR', 'Ukraine', 48.3794, 31.1656, 4, '{0.45, 0.55, 0.4, 0.45}', 0.30, 0.78, '{-0.01, 0.02, -0.01, 0.0}'),
  ('NIC', 'Nicaragua', 12.8654, -85.2072, 3, '{0.25, 0.72, 0.2, 0.28}', 0.28, 0.68, '{-0.01, 0.01, 0.0, 0.0}'),
  ('BDI', 'Burundi', -3.3731, 29.9189, 3, '{0.15, 0.8, 0.12, 0.18}', 0.18, 0.72, '{0.0, 0.0, 0.0, 0.0}'),
  ('CAF', 'Central African Republic', 6.6111, 20.9394, 4, '{0.1, 0.85, 0.08, 0.12}', 0.12, 0.82, '{0.0, 0.0, 0.0, 0.0}'),
  ('TCD', 'Chad', 15.4542, 18.7322, 3, '{0.18, 0.78, 0.15, 0.2}', 0.20, 0.70, '{0.0, 0.0, 0.0, 0.0}'),
  ('COD', 'DR Congo', -4.0383, 21.7587, 4, '{0.15, 0.8, 0.12, 0.18}', 0.16, 0.76, '{0.0, 0.0, 0.0, 0.0}'),
  ('PRK', 'North Korea', 40.3399, 127.5101, 3, '{0.05, 0.98, 0.02, 0.1}', 0.35, 0.62, '{0.0, 0.0, 0.0, 0.0}'),
  ('CUB', 'Cuba', 21.5218, -77.7812, 3, '{0.2, 0.85, 0.15, 0.25}', 0.38, 0.58, '{0.0, 0.0, 0.0, 0.0}'),
  ('BLR', 'Belarus', 53.7098, 27.9534, 3, '{0.25, 0.8, 0.2, 0.3}', 0.40, 0.55, '{0.0, 0.0, 0.0, 0.0}'),

  -- Moderate stress (transition_risk 0.4-0.6)
  ('ARG', 'Argentina', -38.4161, -63.6167, 0, '{0.45, 0.5, 0.4, 0.4}', 0.42, 0.52, '{-0.01, 0.0, 0.0, 0.0}'),
  ('COL', 'Colombia', 4.5709, -74.2973, 0, '{0.48, 0.52, 0.42, 0.45}', 0.48, 0.48, '{0.01, 0.0, 0.0, 0.0}'),
  ('PER', 'Peru', -9.1900, -75.0152, 0, '{0.45, 0.52, 0.4, 0.42}', 0.45, 0.50, '{0.0, 0.0, 0.0, 0.0}'),
  ('PHL', 'Philippines', 12.8797, 121.7740, 0, '{0.5, 0.52, 0.45, 0.48}', 0.50, 0.45, '{0.0, 0.0, 0.0, 0.0}'),
  ('THA', 'Thailand', 15.8700, 100.9925, 4, '{0.45, 0.58, 0.4, 0.45}', 0.48, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('MYS', 'Malaysia', 4.2105, 101.9758, 0, '{0.55, 0.52, 0.5, 0.52}', 0.58, 0.38, '{0.0, 0.0, 0.0, 0.0}'),
  ('BGD', 'Bangladesh', 23.6850, 90.3563, 0, '{0.42, 0.55, 0.38, 0.4}', 0.45, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('IRQ', 'Iraq', 33.2232, 43.6793, 4, '{0.28, 0.68, 0.25, 0.32}', 0.32, 0.62, '{0.01, -0.01, 0.01, 0.0}'),
  ('JOR', 'Jordan', 30.5852, 36.2384, 3, '{0.42, 0.6, 0.38, 0.45}', 0.52, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('MAR', 'Morocco', 31.7917, -7.0926, 3, '{0.45, 0.58, 0.4, 0.48}', 0.55, 0.38, '{0.0, 0.0, 0.0, 0.0}'),
  ('TUN', 'Tunisia', 33.8869, 9.5375, 4, '{0.48, 0.55, 0.42, 0.45}', 0.48, 0.45, '{0.01, 0.0, 0.0, 0.0}'),
  ('DZA', 'Algeria', 28.0339, 1.6596, 3, '{0.38, 0.65, 0.32, 0.4}', 0.45, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('KAZ', 'Kazakhstan', 48.0196, 66.9237, 3, '{0.45, 0.7, 0.38, 0.48}', 0.55, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('UZB', 'Uzbekistan', 41.3775, 64.5853, 3, '{0.35, 0.75, 0.3, 0.38}', 0.48, 0.45, '{0.0, 0.0, 0.0, 0.0}'),
  ('AZE', 'Azerbaijan', 40.1431, 47.5769, 3, '{0.4, 0.72, 0.35, 0.42}', 0.52, 0.45, '{0.0, 0.0, 0.0, 0.0}'),

  -- Stable additions (transition_risk < 0.3)
  ('CAN', 'Canada', 56.1304, -106.3468, 0, '{0.82, 0.28, 0.75, 0.7}', 0.88, 0.08, '{0.0, 0.0, 0.0, 0.0}'),
  ('NZL', 'New Zealand', -40.9006, 174.8860, 0, '{0.85, 0.25, 0.78, 0.72}', 0.92, 0.05, '{0.0, 0.0, 0.0, 0.0}'),
  ('NOR', 'Norway', 60.4720, 8.4689, 2, '{0.88, 0.22, 0.82, 0.75}', 0.94, 0.04, '{0.0, 0.0, 0.0, 0.0}'),
  ('SWE', 'Sweden', 60.1282, 18.6435, 2, '{0.85, 0.25, 0.8, 0.72}', 0.92, 0.06, '{0.0, 0.0, 0.0, 0.0}'),
  ('CHE', 'Switzerland', 46.8182, 8.2275, 0, '{0.9, 0.2, 0.85, 0.78}', 0.95, 0.03, '{0.0, 0.0, 0.0, 0.0}'),
  ('NLD', 'Netherlands', 52.1326, 5.2913, 0, '{0.82, 0.28, 0.75, 0.7}', 0.88, 0.08, '{0.0, 0.0, 0.0, 0.0}'),
  ('BEL', 'Belgium', 50.5039, 4.4699, 0, '{0.78, 0.32, 0.72, 0.68}', 0.82, 0.12, '{0.0, 0.0, 0.0, 0.0}'),
  ('AUT', 'Austria', 47.5162, 14.5501, 0, '{0.8, 0.3, 0.74, 0.7}', 0.85, 0.10, '{0.0, 0.0, 0.0, 0.0}'),
  ('POL', 'Poland', 51.9194, 19.1451, 0, '{0.68, 0.42, 0.62, 0.58}', 0.72, 0.22, '{0.0, 0.0, 0.0, 0.0}'),
  ('CZE', 'Czechia', 49.8175, 15.4730, 0, '{0.72, 0.38, 0.66, 0.62}', 0.78, 0.18, '{0.0, 0.0, 0.0, 0.0}'),
  ('HUN', 'Hungary', 47.1625, 19.5033, 0, '{0.58, 0.52, 0.52, 0.55}', 0.62, 0.32, '{-0.01, 0.01, 0.0, 0.0}'),
  ('ROU', 'Romania', 45.9432, 24.9668, 0, '{0.6, 0.48, 0.55, 0.52}', 0.65, 0.28, '{0.0, 0.0, 0.0, 0.0}'),
  ('GRC', 'Greece', 39.0742, 21.8243, 0, '{0.62, 0.45, 0.58, 0.55}', 0.68, 0.28, '{0.0, 0.0, 0.0, 0.0}'),
  ('PRT', 'Portugal', 39.3999, -8.2245, 0, '{0.72, 0.35, 0.68, 0.62}', 0.78, 0.15, '{0.0, 0.0, 0.0, 0.0}'),
  ('ESP', 'Spain', 40.4637, -3.7492, 0, '{0.7, 0.38, 0.65, 0.6}', 0.75, 0.18, '{0.0, 0.0, 0.0, 0.0}'),
  ('ITA', 'Italy', 41.8719, 12.5674, 0, '{0.68, 0.42, 0.62, 0.58}', 0.72, 0.22, '{0.0, 0.0, 0.0, 0.0}'),
  ('IRL', 'Ireland', 53.1424, -7.6921, 0, '{0.78, 0.32, 0.72, 0.68}', 0.82, 0.12, '{0.0, 0.0, 0.0, 0.0}'),
  ('FIN', 'Finland', 61.9241, 25.7482, 0, '{0.85, 0.25, 0.78, 0.72}', 0.90, 0.08, '{0.0, 0.0, 0.0, 0.0}'),
  ('DNK', 'Denmark', 56.2639, 9.5018, 0, '{0.85, 0.25, 0.78, 0.72}', 0.90, 0.07, '{0.0, 0.0, 0.0, 0.0}'),
  ('SGP', 'Singapore', 1.3521, 103.8198, 0, '{0.75, 0.55, 0.7, 0.72}', 0.85, 0.12, '{0.0, 0.0, 0.0, 0.0}'),
  ('TWN', 'Taiwan', 23.6978, 120.9605, 0, '{0.72, 0.48, 0.68, 0.65}', 0.75, 0.35, '{0.0, 0.01, 0.0, 0.0}'),
  ('VNM', 'Vietnam', 14.0583, 108.2772, 1, '{0.45, 0.72, 0.4, 0.52}', 0.65, 0.32, '{0.01, 0.0, 0.0, 0.0}'),
  ('ARE', 'UAE', 23.4241, 53.8478, 3, '{0.55, 0.68, 0.48, 0.58}', 0.72, 0.22, '{0.0, 0.0, 0.0, 0.0}'),
  ('QAT', 'Qatar', 25.3548, 51.1839, 3, '{0.58, 0.65, 0.5, 0.6}', 0.75, 0.18, '{0.0, 0.0, 0.0, 0.0}'),
  ('KWT', 'Kuwait', 29.3117, 47.4818, 3, '{0.52, 0.62, 0.45, 0.55}', 0.68, 0.25, '{0.0, 0.0, 0.0, 0.0}'),
  ('OMN', 'Oman', 21.4735, 55.9754, 3, '{0.48, 0.65, 0.42, 0.52}', 0.65, 0.28, '{0.0, 0.0, 0.0, 0.0}'),
  ('BHR', 'Bahrain', 26.0667, 50.5577, 3, '{0.45, 0.68, 0.4, 0.5}', 0.58, 0.38, '{0.0, 0.0, 0.0, 0.0}'),
  ('KEN', 'Kenya', -0.0236, 37.9062, 0, '{0.45, 0.52, 0.4, 0.42}', 0.48, 0.45, '{0.0, 0.0, 0.0, 0.0}'),
  ('GHA', 'Ghana', 7.9465, -1.0232, 0, '{0.52, 0.45, 0.48, 0.45}', 0.58, 0.35, '{0.0, 0.0, 0.0, 0.0}'),
  ('SEN', 'Senegal', 14.4974, -14.4524, 0, '{0.5, 0.48, 0.45, 0.45}', 0.55, 0.38, '{0.0, 0.0, 0.0, 0.0}'),
  ('CIV', 'Ivory Coast', 7.5400, -5.5471, 0, '{0.42, 0.55, 0.38, 0.4}', 0.45, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('CMR', 'Cameroon', 7.3697, 12.3547, 3, '{0.35, 0.65, 0.3, 0.38}', 0.40, 0.55, '{0.0, 0.0, 0.0, 0.0}'),
  ('AGO', 'Angola', -11.2027, 17.8739, 3, '{0.38, 0.68, 0.32, 0.4}', 0.45, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('MOZ', 'Mozambique', -18.6657, 35.5296, 0, '{0.35, 0.58, 0.3, 0.35}', 0.38, 0.55, '{0.0, 0.0, 0.0, 0.0}'),
  ('ZWE', 'Zimbabwe', -19.0154, 29.1549, 3, '{0.28, 0.72, 0.22, 0.3}', 0.32, 0.62, '{0.0, 0.0, 0.0, 0.0}'),
  ('TZA', 'Tanzania', -6.3690, 34.8888, 0, '{0.45, 0.52, 0.4, 0.42}', 0.50, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('UGA', 'Uganda', 1.3733, 32.2903, 3, '{0.38, 0.62, 0.32, 0.38}', 0.42, 0.52, '{0.0, 0.0, 0.0, 0.0}'),
  ('RWA', 'Rwanda', -1.9403, 29.8739, 3, '{0.45, 0.7, 0.38, 0.48}', 0.55, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('NPL', 'Nepal', 28.3949, 84.1240, 0, '{0.4, 0.52, 0.35, 0.38}', 0.42, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('LKA', 'Sri Lanka', 7.8731, 80.7718, 0, '{0.45, 0.55, 0.4, 0.45}', 0.48, 0.52, '{-0.01, 0.01, 0.0, 0.0}'),
  ('MMK', 'North Macedonia', 41.5124, 21.7453, 0, '{0.52, 0.48, 0.48, 0.48}', 0.55, 0.38, '{0.0, 0.0, 0.0, 0.0}'),
  ('SRB', 'Serbia', 44.0165, 21.0059, 0, '{0.5, 0.52, 0.45, 0.48}', 0.52, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('BIH', 'Bosnia', 43.9159, 17.6791, 0, '{0.45, 0.55, 0.4, 0.45}', 0.48, 0.48, '{0.0, 0.0, 0.0, 0.0}'),
  ('ALB', 'Albania', 41.1533, 20.1683, 0, '{0.48, 0.52, 0.42, 0.45}', 0.52, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('MDA', 'Moldova', 47.4116, 28.3699, 0, '{0.45, 0.55, 0.4, 0.45}', 0.48, 0.52, '{0.0, 0.0, 0.0, 0.0}'),
  ('GEO', 'Georgia', 42.3154, 43.3569, 0, '{0.52, 0.52, 0.48, 0.5}', 0.55, 0.45, '{0.0, 0.0, 0.0, 0.0}'),
  ('ARM', 'Armenia', 40.0691, 45.0382, 0, '{0.48, 0.55, 0.42, 0.48}', 0.50, 0.48, '{0.0, 0.01, 0.0, 0.0}'),
  ('ECU', 'Ecuador', -1.8312, -78.1834, 0, '{0.42, 0.52, 0.38, 0.4}', 0.45, 0.52, '{0.0, 0.0, 0.0, 0.0}'),
  ('BOL', 'Bolivia', -16.2902, -63.5887, 0, '{0.38, 0.55, 0.35, 0.38}', 0.42, 0.52, '{0.0, 0.0, 0.0, 0.0}'),
  ('PRY', 'Paraguay', -23.4425, -58.4438, 0, '{0.45, 0.5, 0.4, 0.42}', 0.48, 0.45, '{0.0, 0.0, 0.0, 0.0}'),
  ('URY', 'Uruguay', -32.5228, -55.7658, 0, '{0.68, 0.38, 0.62, 0.58}', 0.72, 0.22, '{0.0, 0.0, 0.0, 0.0}'),
  ('CHL', 'Chile', -35.6751, -71.5430, 0, '{0.65, 0.42, 0.58, 0.55}', 0.68, 0.28, '{0.0, 0.0, 0.0, 0.0}'),
  ('CRI', 'Costa Rica', 9.7489, -83.7534, 0, '{0.7, 0.35, 0.65, 0.6}', 0.75, 0.18, '{0.0, 0.0, 0.0, 0.0}'),
  ('PAN', 'Panama', 8.5380, -80.7821, 0, '{0.58, 0.45, 0.52, 0.52}', 0.62, 0.32, '{0.0, 0.0, 0.0, 0.0}'),
  ('GTM', 'Guatemala', 15.7835, -90.2308, 0, '{0.38, 0.55, 0.32, 0.38}', 0.42, 0.52, '{0.0, 0.0, 0.0, 0.0}'),
  ('HND', 'Honduras', 15.2000, -86.2419, 0, '{0.35, 0.58, 0.3, 0.35}', 0.38, 0.58, '{0.0, 0.0, 0.0, 0.0}'),
  ('SLV', 'El Salvador', 13.7942, -88.8965, 0, '{0.4, 0.55, 0.35, 0.4}', 0.45, 0.52, '{0.01, 0.0, 0.0, 0.0}'),
  ('DOM', 'Dominican Republic', 18.7357, -70.1627, 0, '{0.48, 0.5, 0.42, 0.45}', 0.52, 0.42, '{0.0, 0.0, 0.0, 0.0}'),
  ('JAM', 'Jamaica', 18.1096, -77.2975, 0, '{0.5, 0.48, 0.45, 0.48}', 0.55, 0.38, '{0.0, 0.0, 0.0, 0.0}'),
  ('TTO', 'Trinidad and Tobago', 10.6918, -61.2225, 0, '{0.55, 0.45, 0.5, 0.5}', 0.58, 0.35, '{0.0, 0.0, 0.0, 0.0}')

ON CONFLICT (code) DO UPDATE SET
  position = EXCLUDED.position,
  basin_strength = EXCLUDED.basin_strength,
  transition_risk = EXCLUDED.transition_risk,
  velocity = EXCLUDED.velocity,
  updated_at = NOW();

-- ============================================
-- CREATE HISTORICAL TRACKING TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS nation_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  nation_code VARCHAR(3) REFERENCES nations(code) ON DELETE CASCADE,
  recorded_at TIMESTAMPTZ DEFAULT NOW(),

  -- State snapshot
  position DOUBLE PRECISION[] NOT NULL,
  velocity DOUBLE PRECISION[] NOT NULL,
  basin_strength DOUBLE PRECISION NOT NULL,
  transition_risk DOUBLE PRECISION NOT NULL,

  -- Computed trends
  basin_strength_7d_delta DOUBLE PRECISION,  -- Change over 7 days
  transition_risk_7d_delta DOUBLE PRECISION,
  velocity_magnitude DOUBLE PRECISION,       -- Speed of movement in phase space

  -- Source signals that contributed
  signal_sources JSONB DEFAULT '{}',

  UNIQUE(nation_code, recorded_at)
);

CREATE INDEX IF NOT EXISTS idx_nation_history_code_time
  ON nation_history(nation_code, recorded_at DESC);

-- Enable RLS
ALTER TABLE nation_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view nation history"
  ON nation_history FOR SELECT USING (true);

CREATE POLICY "Service role can insert nation history"
  ON nation_history FOR INSERT
  WITH CHECK (true);

-- ============================================
-- FUNCTION TO RECORD CURRENT STATE
-- ============================================

CREATE OR REPLACE FUNCTION record_nation_snapshot()
RETURNS void AS $$
BEGIN
  INSERT INTO nation_history (
    nation_code, position, velocity, basin_strength, transition_risk, velocity_magnitude
  )
  SELECT
    code,
    position,
    velocity,
    basin_strength,
    transition_risk,
    SQRT(
      POWER(velocity[1], 2) +
      POWER(velocity[2], 2) +
      POWER(velocity[3], 2) +
      POWER(velocity[4], 2)
    ) as velocity_magnitude
  FROM nations;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- VIEW FOR NATIONS NEAR PHASE TRANSITION
-- ============================================

CREATE OR REPLACE VIEW nations_at_risk AS
SELECT
  code,
  name,
  basin_strength,
  transition_risk,
  regime,
  SQRT(
    POWER(velocity[1], 2) +
    POWER(velocity[2], 2) +
    POWER(velocity[3], 2) +
    POWER(velocity[4], 2)
  ) as velocity_magnitude,
  CASE
    WHEN transition_risk > 0.8 THEN 'CRITICAL'
    WHEN transition_risk > 0.6 THEN 'HIGH'
    WHEN transition_risk > 0.4 THEN 'ELEVATED'
    WHEN transition_risk > 0.2 THEN 'MODERATE'
    ELSE 'LOW'
  END as risk_level
FROM nations
WHERE transition_risk > 0.4 OR basin_strength < 0.4
ORDER BY transition_risk DESC, basin_strength ASC;

-- ============================================
-- VIEW FOR TREND ANALYSIS
-- ============================================

CREATE OR REPLACE VIEW nation_trends AS
SELECT
  n.code,
  n.name,
  n.basin_strength as current_basin,
  n.transition_risk as current_risk,
  h7.basin_strength as basin_7d_ago,
  h7.transition_risk as risk_7d_ago,
  h30.basin_strength as basin_30d_ago,
  h30.transition_risk as risk_30d_ago,
  (n.basin_strength - COALESCE(h7.basin_strength, n.basin_strength)) as basin_7d_delta,
  (n.transition_risk - COALESCE(h7.transition_risk, n.transition_risk)) as risk_7d_delta,
  (n.basin_strength - COALESCE(h30.basin_strength, n.basin_strength)) as basin_30d_delta,
  (n.transition_risk - COALESCE(h30.transition_risk, n.transition_risk)) as risk_30d_delta,
  CASE
    WHEN (n.transition_risk - COALESCE(h7.transition_risk, n.transition_risk)) > 0.1 THEN 'DETERIORATING'
    WHEN (n.transition_risk - COALESCE(h7.transition_risk, n.transition_risk)) < -0.1 THEN 'IMPROVING'
    ELSE 'STABLE'
  END as trajectory
FROM nations n
LEFT JOIN LATERAL (
  SELECT basin_strength, transition_risk
  FROM nation_history
  WHERE nation_code = n.code
    AND recorded_at >= NOW() - INTERVAL '7 days'
  ORDER BY recorded_at ASC
  LIMIT 1
) h7 ON true
LEFT JOIN LATERAL (
  SELECT basin_strength, transition_risk
  FROM nation_history
  WHERE nation_code = n.code
    AND recorded_at >= NOW() - INTERVAL '30 days'
  ORDER BY recorded_at ASC
  LIMIT 1
) h30 ON true;
