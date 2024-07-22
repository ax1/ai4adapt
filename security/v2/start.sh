#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
file="AI4ADAPT_REPORT_${timestamp}.txt"
echo $file
python3 -u 3-using_PPO.py > "$file"