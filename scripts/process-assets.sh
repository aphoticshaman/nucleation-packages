#!/bin/bash
# LatticeForge Asset Processor
# Copies and organizes art assets from art_assets/ to packages/web/public/images/
# with clean, web-friendly names for Vercel edge CDN delivery

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/art_assets"
DEST="$REPO_ROOT/packages/web/public/images"

echo "=== LatticeForge Asset Processor ==="
echo "Source: $SRC"
echo "Destination: $DEST"
echo ""

# Create directory structure
mkdir -p "$DEST"/{bg,hero,network,shapes,dashboard,brand,icons,badges,states,onboarding,atmosphere,social,risk,video}

# === BACKGROUNDS & TEXTURES ===
echo "Processing backgrounds..."
cp "$SRC/aphoticshaman_Abstract_dark_granite_or_obsidian_surface_with__b88ef790-9508-49e6-81dc-5be6729546f0_0.png" "$DEST/bg/obsidian.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Abstract_dark_slate_blue_gradient_with_faint_he_97441d10-208b-49c1-824d-24421c23b8bf_1.png" "$DEST/bg/slate-gradient.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Abstract_gradient_deep_navy_bottom_transitionin_b94719ff-dde3-415f-b235-a8424133d864_0.png" "$DEST/bg/navy-gradient.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Extreme_macro_of_dark_brushed_titanium_with_fai_3e1a9e65-7060-4b3a-a324-1d5d054b2bb7_0.png" "$DEST/bg/titanium-hex.png" 2>/dev/null || true

# === GLOBE & GEOSPATIAL ===
echo "Processing globe/geospatial..."
cp "$SRC/LATTICE FORGE GLOBE WIRE NEEDS TEXT CROPPED NOW.png" "$DEST/hero/globe-wire.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_3D_geospatial_intelligence_map_hexagonal_bins_e_2b217a49-d83d-4d4a-abcc-6f8543192df1_0.png" "$DEST/hero/hexbin-map.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Wireframe_sphere_with_latitude_longitude_lines__9479272e-e719-41e7-ad99-d0d939f63bfc_3.png" "$DEST/hero/wireframe-sphere.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimalist_dark_blue_globe_with_glowing_data_co_12065a96-2689-4179-9959-9bfa0a826fbe_0.png" "$DEST/hero/data-globe.png" 2>/dev/null || true
cp "$SRC/earth.png" "$DEST/hero/earth.png" 2>/dev/null || true

# === NETWORK & NODES ===
echo "Processing network visuals..."
cp "$SRC/aphoticshaman_Abstract_visualization_of_interconnected_nodes__dcc2e015-bd29-418a-956d-2f3f2270f31d_1.png" "$DEST/network/nodes.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Geometric_constellation_of_connected_points_and_bb301e59-d2b0-4475-9570-df46b10ec2a4_0.png" "$DEST/network/constellation.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Small_cluster_of_5-7_connected_glowing_nodes_th_a3d3d134-5889-4056-af46-a42ed1e421c0_0.png" "$DEST/network/cluster.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_node-link_causality_diagram_flowing_from_left_t_0e06a24d-b591-42df-a7b3-bbd296c6a9f0_2.png" "$DEST/network/causality.png" 2>/dev/null || true

# === ABSTRACT SHAPES ===
echo "Processing abstract shapes..."
cp "$SRC/aphoticshaman_Circular_ring_with_traveling_light_pulse_neon_b_13cadba0-bf69-40ce-9553-aa219c6f8e04_0.png" "$DEST/shapes/ring-pulse.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Single_hexagon_outline_thin_glowing_cyan_wirefr_8ec0f195-1aeb-44bd-bd33-eded9eefc1ec_0.png" "$DEST/shapes/hexagon.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Single_perfect_sphere_floating_in_void_surface__1d1ee918-686f-432c-a69d-96570055a679_0.png" "$DEST/shapes/sphere.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Single_floating_crystal_shard_blue-purple_gradi_8457cd01-3c8b-424a-93a0-ea7fb6685862_1.png" "$DEST/shapes/crystal.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Single_glowing_blue_energy_sphere_soft_plasma_c_f28ecd1e-4fc2-417c-8f1a-2ea56b7058b0_0.png" "$DEST/shapes/plasma.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Infinity_symbol_made_of_flowing_light_particles_786821d7-4605-4582-869c-5e29fe675337_1.png" "$DEST/shapes/infinity.png" 2>/dev/null || true

# === DASHBOARD & ANALYTICS ===
echo "Processing dashboard assets..."
cp "$SRC/aphoticshaman_futuristic_intelligence_analyst_dashboard_dark__2457a741-cdb9-4e88-9fb0-ced7810e2714_0.png" "$DEST/dashboard/mockup.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_horizontal_waterfall_chart_red_bars_pushing_rig_733aad1e-5228-4e85-a761-c3d03df3a58f_0.png" "$DEST/dashboard/waterfall.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Blurred_out-of-focus_view_of_multiple_monitors__4236c985-e23a-4e3d-b223-acc33478c0c1_1.png" "$DEST/dashboard/monitors.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Defocused_background_of_war_room_with_multiple__32a03fc8-2ff9-4b13-89cb-2b8c8017084a_0.png" "$DEST/dashboard/war-room.png" 2>/dev/null || true

# === LOGOS & BRANDING ===
echo "Processing brand assets..."
cp "$SRC/aphoticshaman_App_icon_for_LatticeForge_intelligence_platform_fd831ec4-b981-45c0-b01e-5a64eaf2f52c_2.png" "$DEST/brand/app-icon.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Professional_logo_design_for_LatticeForge_a_pre_7d06131e-aff3-45a1-a903-701da1ec9d0f_3.png" "$DEST/brand/logo.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Sophisticated_monogram_logo_combining_the_lette_1e62d1f1-daa8-4bff-8e7b-51d4c91114ae_0.png" "$DEST/brand/monogram.png" 2>/dev/null || true

# === FEATURE ICONS ===
echo "Processing feature icons..."
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_analytics._Three_ascen_474719af-bdf2-4179-beef-a40fdf0c6953_0.png" "$DEST/icons/analytics.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_API_access._Two_geomet_d5715fca-1473-416d-b78d-fcc897358281_0.png" "$DEST/icons/api.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_data_export._Upward_ar_2fb70048-46a4-4187-94a0-6d48f3a0ecde_3.png" "$DEST/icons/export.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_security_features._Geo_0028c1a1-5ae8-41b3-a7d0-b5b90107e7a1_2.png" "$DEST/icons/security.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_simulations._Small_glo_cfa4a142-e03a-4e7e-9dd2-df87af6cf5ed_0.png" "$DEST/icons/simulation.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_team_features._Three_s_c4626d80-28da-4aab-a026-0b9fbb35762b_2.png" "$DEST/icons/team.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_feature_icon_for_webhooks._Stylized_hoo_95c193cd-4c29-4b1c-bb2d-9a2bd126570c_3.png" "$DEST/icons/webhook.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Three_ascending_vertical_bars_with_glow_effect__8827cf34-bef9-4979-9a74-d58c180cd6a0_1.png" "$DEST/icons/growth.png" 2>/dev/null || true

# === TIER BADGES ===
echo "Processing tier badges..."
cp "$SRC/aphoticshaman_Minimal_tier_badge_icon_for_Trial_plan._Simple__96007383-2c4a-4d93-a416-b3f87998d8c7_0.png" "$DEST/badges/trial.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Tier_badge_icon_for_Starter_plan._Hexagonal_fra_84127c06-9aa3-4d65-81f2-2065d8a4b0df_0.png" "$DEST/badges/starter.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Tier_badge_icon_for_Pro_plan._Hexagonal_frame_w_c9c05172-6283-4db8-ae1f-373da3dafc42_0.png" "$DEST/badges/pro.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Tier_badge_icon_for_Enterprise_plan._Shield_sha_61a152b5-e164-4411-b5e5-b0a11fdefba9_0.png" "$DEST/badges/enterprise.png" 2>/dev/null || true

# === ERROR & EMPTY STATES ===
echo "Processing state illustrations..."
cp "$SRC/aphoticshaman_Minimal_illustration_for_404_error_page._Single_6a7ade31-8bf2-4d4f-8c15-312889cbf662_1.png" "$DEST/states/404.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_illustration_for_500_server_error._Forg_17275013-c7ee-4619-b3b2-d4c9a1faf579_2.png" "$DEST/states/500.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_illustration_for_connection_error._Two__b8321b99-e03a-4171-acba-013ec3587e02_0.png" "$DEST/states/connection-error.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_illustration_for_empty_simulations_list_80c7a7e6-43ac-4ee5-b91c-d20e7766ba1d_2.png" "$DEST/states/empty.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_illustration_for_success_celebration._C_392321a1-a0cf-43be-b609-9520a899ff3b_1.png" "$DEST/states/success.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimalist_illustration_for_dashboard_empty_sta_08e2e5a1-08ab-4c12-bd63-dcc149eca838_2.png" "$DEST/states/empty-dashboard.png" 2>/dev/null || true

# === ONBOARDING ===
echo "Processing onboarding..."
cp "$SRC/aphoticshaman_Abstract_visualization_for_onboarding_analysis__8c79953c-5516-4a21-b74d-435e8e575775_1.png" "$DEST/onboarding/analysis.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Abstract_visualization_for_onboarding_connectin_7c0adf66-50e1-44d9-b6cc-b1ee162400aa_0.png" "$DEST/onboarding/connecting.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Abstract_visualization_for_onboarding_exporting_f17b78f4-129f-4fe6-874f-410519ac7af2_1.png" "$DEST/onboarding/exporting.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Abstract_visualization_for_onboarding_simulatio_a9d7d9a9-c4d4-4bd7-92d7-348bc55dc8b6_0.png" "$DEST/onboarding/simulation.png" 2>/dev/null || true

# === ATMOSPHERIC ===
echo "Processing atmospheric..."
cp "$SRC/aphoticshaman_Aerial_view_of_calm_deep_blue_ocean_with_a_sing_e6412aef-0e7a-4510-9b5f-5165ab736063_1.png" "$DEST/atmosphere/ocean.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Aerial_view_of_deep_circular_sinkhole_in_ocean__6a3d79ff-1908-4c85-ab90-f313ad275349_0.png" "$DEST/atmosphere/sinkhole.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Single_crystal_chess_piece_king_on_dark_reflect_3ad635cc-d3ca-48a0-a9d9-5dd2da405833_1.png" "$DEST/atmosphere/chess-king.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Mercury_droplet_shaped_like_earth_hovering_refl_48219cf9-6e36-409b-8ab4-ef17303b2388_0.png" "$DEST/atmosphere/mercury-earth.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Top-down_view_of_dark_wooden_table_with_scatter_417a7551-974c-4e49-90a0-aa4dd87e2d48_0.png" "$DEST/atmosphere/workspace.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimalist_landscape_dark_foreground_silhouette_1c83b7fd-d707-4632-b33d-c0cf03e4b772_1.png" "$DEST/atmosphere/horizon.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Abstract_topographic_map_with_glowing_contour_l_3720ecaa-558c-4621-bc89-d64f45d600d2_1.png" "$DEST/atmosphere/topographic.png" 2>/dev/null || true

# === SOCIAL MEDIA ===
echo "Processing social media..."
cp "$SRC/aphoticshaman_Social_media_card_graphic_for_LatticeForge_prod_4f7650f6-7e98-4987-9f85-6c536699d2f5_0.png" "$DEST/social/twitter-card.png" 2>/dev/null || true

# === RISK & ALERT ===
echo "Processing risk visuals..."
cp "$SRC/aphoticshaman_Dark_background_with_glowing_orange_and_red_ene_5b5d5576-9584-41cb-82ea-b24e94c166fb_0.png" "$DEST/risk/energy.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_geometric_shield_shape_chrome_metallic__2517918e-b3aa-46cf-8786-5b2f53d83a40_1.png" "$DEST/risk/shield.png" 2>/dev/null || true
cp "$SRC/aphoticshaman_Minimal_L-shaped_corner_bracket_thin_glowing_li_8c4f5978-8e2f-4e17-8352-6bb14b0a55ba_0.png" "$DEST/risk/bracket.png" 2>/dev/null || true

# === VIDEOS ===
echo "Processing videos..."
cp "$SRC/grok-video-b5956bdf-38c4-4c53-a43a-1d42287e058d.mp4" "$DEST/video/hero-1.mp4" 2>/dev/null || true
cp "$SRC/grok-video-e509f9e8-bdd3-49dc-9c35-1f9102c6f59d.mp4" "$DEST/video/hero-2.mp4" 2>/dev/null || true

echo ""
echo "=== Asset Processing Complete ==="
echo ""
echo "Assets copied to: $DEST"
find "$DEST" -type f | wc -l | xargs echo "Total files:"
du -sh "$DEST" | cut -f1 | xargs echo "Total size:"
echo ""
echo "Assets will be served from Vercel's global edge CDN."
echo "Use 'import { assets } from \"@/lib/assets\"' in components."
