#!/usr/bin/env python3
"""
Test script demonstrating the behavior thresholds feature in Biwipy-proto.

This shows:
1. How profiles are stored with optional behavior thresholds
2. How thresholds are applied to CyclistBehavior
3. Example JSON schema for profiles
"""

import json
from pathlib import Path

# Example profile with behavior thresholds
profile_with_thresholds = {
    "name": "Pro Descender",
    "CdA": 0.45,
    "Cr": 0.004,
    "mass_kg": 75.0,
    "profile_preset": "pro",
    "uphill_mode": "aggressive",
    "downhill_mode": "aggressive",
    "corner_mode": "aggressive",
    "behavior_thresholds": {
        "uphill_facteur_forte": 5.0,  # More aggressive uphill
        "downhill_vitesse_max_absolue": 22.0,  # Higher max downhill speed (m/s)
        "corner_speed_slight": 20.0,  # Higher corner speed (m/s)
        "corner_speed_straight": 24.0,
    },
    "updated_at": "2026-07-02T14:30:00+00:00"
}

# Example profile without thresholds (basic)
profile_basic = {
    "name": "Conservative Rider",
    "CdA": 0.50,
    "Cr": 0.005,
    "mass_kg": 85.0,
    "profile_preset": "conservative",
    "uphill_mode": "conservative",
    "downhill_mode": "conservative",
    "corner_mode": "conservative",
    "updated_at": "2026-07-02T12:00:00+00:00"
    # No behavior_thresholds key → uses preset defaults
}

print("=" * 70)
print("PROFILE SCHEMA WITH BEHAVIOR THRESHOLDS")
print("=" * 70)
print("\n1. PROFILE WITH CUSTOMIZED THRESHOLDS (saved to JSON):\n")
print(json.dumps(profile_with_thresholds, indent=2, ensure_ascii=False))

print("\n" + "-" * 70)
print("\n2. PROFILE WITHOUT THRESHOLDS (basic, saved to JSON):\n")
print(json.dumps(profile_basic, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("HOW THE APP HANDLES THRESHOLDS")
print("=" * 70)

print("""
When loading a profile in Biwipy-proto:

1. LOAD PHASE:
   • _load_cyclist_profile_content() reads JSON file
   • _sanitize_profile_form() validates thresholds (only valid keys, numeric values)
   • profile_form['behavior_thresholds'] dict is populated

2. SAVE PHASE:
   • save_cyclist_profile_file() writes behavior_thresholds to JSON if non-empty
   • Empty thresholds are omitted from JSON (keeps file clean)

3. SIMULATION PHASE:
   • build_behavior() creates CyclistBehavior from preset
   • _apply_behavior_thresholds() updates behavior attributes from profile
   • Each threshold value is clamped to valid range (safety)

EXAMPLE THRESHOLDS APPLICATION:
   behavior = CyclistBehavior()  # e.g., 'pro' preset
   _apply_behavior_thresholds(behavior, {
       "downhill_vitesse_max_absolute": 22.0,
       "corner_speed_slight": 20.0
   })
   # Result: behavior.downhill_vitesse_max_absolue = 22.0 (clamped if needed)
   #         behavior.corner_speed_slight = 20.0
   #         Other attributes keep preset defaults
""")

print("\n" + "=" * 70)
print("VALID THRESHOLD PARAMETERS (Customizable)")
print("=" * 70)

threshold_categories = {
    "UPHILL FACTORS": [
        "uphill_facteur_forte (range: 1.0–10.0)",
        "uphill_facteur_moderee (range: 1.0–10.0)",
        "uphill_facteur_legere (range: 1.0–10.0)",
    ],
    "DOWNHILL PARAMETERS": [
        "downhill_vitesse_max_absolue [m/s] (range: 5.0–30.0)",
        "downhill_vitesse_reduction_factor (range: 0.5–10.0)",
        "downhill_vitesse_reduction_cap (range: 0.0–1.0)",
        "downhill_puissance_min [W] (range: 1–50)",
        "downhill_facteur_forte (range: 1.0–50.0)",
        "downhill_facteur_legere (range: 1.0–50.0)",
        "downhill_corner_safety_factor (range: 0.5–1.0)",
    ],
    "CORNER SPEEDS [m/s]": [
        "corner_speed_straight (range: 5.0–30.0)",
        "corner_speed_slight (range: 5.0–30.0)",
        "corner_speed_moderate (range: 5.0–30.0)",
        "corner_speed_sharp (range: 1.0–20.0)",
        "corner_speed_hairpin (range: 1.0–10.0)",
    ]
}

for category, params in threshold_categories.items():
    print(f"\n{category}:")
    for param in params:
        print(f"  • {param}")

print("\n" + "=" * 70)
print("UI FLOW IN APP")
print("=" * 70)

print("""
PROFILE MANAGER SCREEN (dedicated_screen):
  ├─ Profile Selection (load/edit/delete files)
  ├─ Profile Editor (name, CdA, Cr, mass_kg)
  ├─ Basic Behavior (preset, uphill/downhill/corner modes)
  └─ EXPANSION: "Advanced Thresholds (optional)"
      ├─ Uphill Factors
      │  ├─ Facteur Forte
      │  ├─ Facteur Moderee
      │  └─ Facteur Legere
      ├─ Downhill Parameters
      │  ├─ Max Speed (absolute)
      │  ├─ Reduction Factor
      │  ├─ Reduction Cap
      │  ├─ Min Power
      │  ├─ Facteur Forte
      │  ├─ Facteur Legere
      │  └─ Corner Safety Factor
      └─ Corner Speeds
         ├─ Straight Line
         ├─ Slight Turn
         ├─ Moderate Turn
         ├─ Sharp Turn
         └─ Hairpin

SAVE: Saves profile_form (including behavior_thresholds) to JSON
LOAD: Loads profile JSON → applies thresholds to simulation behavior
""")

print("\n✓ Threshold system is fully optional:")
print("  • Profiles without thresholds: use preset defaults")
print("  • Profiles with thresholds: override defaults for simulation")
print("  • UI is clean: advanced panel is collapsed by default")
