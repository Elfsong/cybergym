#!/bin/bash

# Display help message if no arguments provided or help flag used
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ -z "$1" ]; then
    echo "Usage: $0 <poc_file_path>"
    echo ""
    echo "Submit a vulnerability proof-of-concept file to the CyberGym server"
    echo ""
    echo "Arguments:"
    echo "  <poc_file_path>    Path to the proof-of-concept file to submit"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/poc"
    exit 0
fi

POC_FILE="$1"

# Check if file exists
if [ ! -f "$POC_FILE" ]; then
    echo "Error: File '$POC_FILE' not found!"
    exit 1
fi

curl -X POST http://172.17.0.1:8666/submit-vul \
  -F 'metadata={"task_id": "arvo:60262", "agent_id": "5896b54ed4314f8cad769137c55208b4", "checksum": "6ccc0a692130dc4886c1ff32004e9712b39eadf64e7148c1467fb058613ba699", "require_flag": false}' \
  -F "file=@${POC_FILE}"