#!/bin/bash

# Championship run.sh - Script to calculate travel reimbursement
# Takes exactly 3 parameters: trip_duration_days, miles_traveled, total_receipts_amount
# Outputs a single number (the reimbursement amount)

# Check if we have the correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Exactly 3 arguments required" >&2
    echo "Usage: $0 <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
    exit 1
fi

# Validate that arguments are numeric
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: trip_duration_days must be a positive integer" >&2
    exit 1
fi

if ! [[ "$2" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: miles_traveled must be a non-negative number" >&2
    exit 1
fi

if ! [[ "$3" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Error: total_receipts_amount must be a non-negative number" >&2
    exit 1
fi

# Execute the Python calculator script with error handling
python3 championship_calculator.py "$1" "$2" "$3" 2>/dev/null
exit_code=$?

# Check if the Python script executed successfully
if [ $exit_code -ne 0 ]; then
    echo "Error: Calculation failed" >&2
    exit $exit_code
fi

