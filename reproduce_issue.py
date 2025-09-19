#!/usr/bin/env python3
"""
Script to reproduce the coordinate truncation issue when loading тест_точности.toml profile.
"""

import sys
sys.path.append('src')

from profiles import load_profile

def main():
    print("Testing coordinate precision issue...")
    
    # Load the profile that has the issue
    profile_name = "тест_точности"
    print(f"Loading profile: {profile_name}")
    
    settings = load_profile(profile_name)
    
    print(f"Original TOML values:")
    print(f"  control_point_x = 5421243")
    print(f"  control_point_y = 7448559")
    
    print(f"\nLoaded MapSettings values:")
    print(f"  control_point_x = {settings.control_point_x}")
    print(f"  control_point_y = {settings.control_point_y}")
    
    print(f"\nCalculated SK42 GK coordinates:")
    print(f"  control_point_x_sk42_gk = {settings.control_point_x_sk42_gk}")
    print(f"  control_point_y_sk42_gk = {settings.control_point_y_sk42_gk}")
    
    # Check if the last 3 digits are preserved
    original_x = 5421243
    original_y = 7448559
    
    x_matches = settings.control_point_x == original_x
    y_matches = settings.control_point_y == original_y
    
    print(f"\nPrecision check:")
    print(f"  X coordinate preserved: {x_matches}")
    print(f"  Y coordinate preserved: {y_matches}")
    
    if not x_matches:
        print(f"  X difference: {settings.control_point_x - original_x}")
    if not y_matches:
        print(f"  Y difference: {settings.control_point_y - original_y}")

if __name__ == "__main__":
    main()