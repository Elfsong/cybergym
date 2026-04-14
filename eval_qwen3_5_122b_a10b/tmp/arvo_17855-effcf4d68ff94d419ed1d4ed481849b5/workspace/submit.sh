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
  -F 'metadata={"task_id": "arvo:17855", "agent_id": "effcf4d68ff94d419ed1d4ed481849b5", "checksum": "1ba89eef8ec5d1916bc6b732008d90e2f47cd02cc7fc505b1d7968c6e8075481", "require_flag": false}' \
  -F "file=@${POC_FILE}"