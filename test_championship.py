#!/usr/bin/env python3
"""
Championship Test Script with Diagnostic Tracking
Tests the championship calculator and provides detailed performance analytics
"""

import subprocess
import json
import numpy as np
import time
from collections import defaultdict

def test_championship_calculator():
    """Test the championship calculator with comprehensive diagnostics"""
    
    # Load public cases for testing
    with open('/home/ubuntu/challenge/top-coder-challenge/public_cases.json', 'r') as f:
        public_cases = json.load(f)
    
    print("🏆 CHAMPIONSHIP CALCULATOR TESTING")
    print("=" * 60)
    
    # Initialize tracking variables
    total_error = 0
    exact_matches = 0
    close_matches = 0
    good_matches = 0
    test_count = 0
    errors = []
    
    # Diagnostic categories
    diagnostics = {
        "short_trips": {"errors": [], "count": 0},
        "high_mileage": {"errors": [], "count": 0},
        "high_receipts": {"errors": [], "count": 0},
        "extreme_cases": {"errors": [], "count": 0}
    }
    
    start_time = time.time()
    
    # Test all public cases
    for i, case in enumerate(public_cases):
        input_data = case['input']
        expected = case['expected_output']
        
        duration = input_data['trip_duration_days']
        miles = input_data['miles_traveled']
        receipts = input_data['total_receipts_amount']
        
        # Run the calculator
        cmd = [
            './run.sh',
            str(duration),
            str(miles),
            str(receipts)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/ubuntu/challenge/top-coder-challenge')
            if result.returncode == 0:
                predicted = float(result.stdout.strip())
                error = abs(predicted - expected)
                total_error += error
                test_count += 1
                
                # Categorize accuracy
                if error <= 0.01:
                    exact_matches += 1
                    match_status = "EXACT"
                elif error <= 1.00:
                    close_matches += 1
                    match_status = "CLOSE"
                elif error <= 10.00:
                    good_matches += 1
                    match_status = "GOOD"
                else:
                    match_status = "ERROR"
                    errors.append((i+1, error, input_data, expected, predicted))
                
                # Track diagnostics by category
                if duration <= 3:
                    diagnostics["short_trips"]["errors"].append(error)
                    diagnostics["short_trips"]["count"] += 1
                
                if miles > 1000:
                    diagnostics["high_mileage"]["errors"].append(error)
                    diagnostics["high_mileage"]["count"] += 1
                
                if receipts > 3000:
                    diagnostics["high_receipts"]["errors"].append(error)
                    diagnostics["high_receipts"]["count"] += 1
                
                if receipts > 3000 or miles > 1200 or duration > 12:
                    diagnostics["extreme_cases"]["errors"].append(error)
                    diagnostics["extreme_cases"]["count"] += 1
                
                # Show first 20 cases or any with significant error
                if i < 20 or error > 5.0:
                    print(f"Case {i+1:3d}: Days={duration:2d}, Miles={miles:4d}, Receipts=${receipts:7.2f}")
                    print(f"          Expected: ${expected:7.2f}, Got: ${predicted:7.2f}, Error: ${error:6.2f} - {match_status}")
                    if i < 20:
                        print()
                
            else:
                print(f"Case {i+1}: ERROR - {result.stderr}")
        except Exception as e:
            print(f"Case {i+1}: EXCEPTION - {e}")
    
    end_time = time.time()
    
    # Calculate performance metrics
    if test_count > 0:
        avg_error = total_error / test_count
        exact_pct = (exact_matches / test_count) * 100
        close_pct = (close_matches / test_count) * 100
        good_pct = (good_matches / test_count) * 100
        
        print(f"\n{'='*60}")
        print(f"🎯 CHAMPIONSHIP PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total test cases: {test_count}")
        print(f"Average error: ${avg_error:.2f}")
        print(f"Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
        print(f"Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
        print(f"Good matches (±$10.00): {good_matches} ({good_pct:.1f}%)")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"Estimated leaderboard score: {int(avg_error * 100)}")
        
        # Diagnostic breakdown
        print(f"\n📊 DIAGNOSTIC BREAKDOWN BY CATEGORY")
        print(f"{'='*60}")
        
        for category, data in diagnostics.items():
            if data["count"] > 0:
                cat_avg_error = np.mean(data["errors"])
                cat_max_error = np.max(data["errors"])
                print(f"{category.replace('_', ' ').title()}: {data['count']} cases")
                print(f"  Average error: ${cat_avg_error:.2f}")
                print(f"  Maximum error: ${cat_max_error:.2f}")
                print()
        
        # Show worst performing cases
        if errors:
            print(f"🚨 HIGH ERROR CASES (Top {min(10, len(errors))}):")
            print(f"{'='*60}")
            errors.sort(key=lambda x: x[1], reverse=True)
            for i, (case_num, error, inp, exp, pred) in enumerate(errors[:10]):
                print(f"{i+1:2d}. Case {case_num}: ${error:.2f} error")
                print(f"    Input: {inp['trip_duration_days']}d, {inp['miles_traveled']}mi, ${inp['total_receipts_amount']:.2f}")
                print(f"    Expected: ${exp:.2f}, Got: ${pred:.2f}")
                print()
        
        # Leaderboard prediction
        print(f"🏆 LEADERBOARD PREDICTION")
        print(f"{'='*60}")
        if avg_error < 50:
            print("🥇 LIKELY 1ST PLACE - Exceptional performance!")
        elif avg_error < 100:
            print("🥈 LIKELY TOP 3 - Outstanding performance!")
        elif avg_error < 200:
            print("🥉 LIKELY TOP 10 - Excellent performance!")
        else:
            print("📈 GOOD PERFORMANCE - Room for improvement")
        
        print(f"Score range: {int(avg_error * 100)} (lower is better)")
        print(f"Accuracy: {exact_pct + close_pct:.1f}% within $1.00")
    
    return avg_error if test_count > 0 else float('inf')

if __name__ == "__main__":
    test_championship_calculator()

