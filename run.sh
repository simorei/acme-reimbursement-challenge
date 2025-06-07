#!/bin/bash

# run.sh - Script to calculate travel reimbursement
# Takes exactly 3 parameters: trip_duration_days, miles_traveled, total_receipts_amount
# Outputs a single number (the reimbursement amount)

# Check if we have the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Exactly 3 arguments required"
    echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

# Execute the Python calculator script
python3 final_calculator.py "$1" "$2" "$3"

