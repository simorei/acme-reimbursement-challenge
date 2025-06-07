#!/usr/bin/env python3
"""
Final Legacy Travel Reimbursement System Replica
Version 4 - Addressing long trip issues
"""

import sys
import math

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Final reimbursement calculation addressing all identified issues
    """
    
    # Base calculation for different trip durations
    if trip_duration_days <= 3:
        # Short trips
        base_amount = 300 + (trip_duration_days - 1) * 50
    elif trip_duration_days <= 7:
        # Medium trips
        base_amount = 400 + (trip_duration_days - 3) * 100
    elif trip_duration_days <= 10:
        # Long trips
        base_amount = 800 + (trip_duration_days - 7) * 80
    else:
        # Very long trips (capped growth)
        base_amount = 1040 + (trip_duration_days - 10) * 50
    
    # Mileage component with tiers
    if miles_traveled <= 100:
        mileage_rate = 1.0
    elif miles_traveled <= 300:
        mileage_rate = 0.8
    elif miles_traveled <= 600:
        mileage_rate = 0.6
    else:
        mileage_rate = 0.4
    
    mileage_component = miles_traveled * mileage_rate
    
    # Receipt component with diminishing returns
    if total_receipts_amount <= 100:
        receipt_multiplier = 0.5
    elif total_receipts_amount <= 500:
        receipt_multiplier = 0.3
    elif total_receipts_amount <= 1000:
        receipt_multiplier = 0.2
    else:
        receipt_multiplier = 0.1
    
    receipt_component = total_receipts_amount * receipt_multiplier
    
    # Combine components
    reimbursement = base_amount + mileage_component + receipt_component
    
    # Efficiency adjustments
    miles_per_day = miles_traveled / trip_duration_days
    if miles_per_day > 300:
        reimbursement *= 0.8  # Penalty for very high efficiency
    elif miles_per_day < 30:
        reimbursement *= 0.9  # Penalty for very low efficiency
    
    # Receipt per day adjustments
    receipts_per_day = total_receipts_amount / trip_duration_days
    if receipts_per_day > 200:
        reimbursement *= 0.8  # Penalty for very high daily spending
    
    # Special case for 5-day trips (mentioned in interviews)
    if trip_duration_days == 5:
        reimbursement *= 1.05
    
    # Special case for very long trips with high receipts
    if trip_duration_days >= 12 and total_receipts_amount > 300:
        reimbursement = min(reimbursement, 1500)  # Cap reimbursement
    
    return reimbursement

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 4:
        print("Usage: python3 final_calculator.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = int(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print("Error: Invalid input format")
        sys.exit(1)
    
    reimbursement = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Round to 2 decimal places as required
    print(f"{reimbursement:.2f}")

if __name__ == "__main__":
    main()

