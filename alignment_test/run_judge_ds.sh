#!/bin/bash

# Example script to run LLM as Judge for rule-case relevance evaluation

# Set API key (replace with your actual key)
export DEEPSEEK_API_KEY=''


DATA_DIR="../data/cases"
OUTPUT_BASE_DIR="./alignment_test/alignment_results"


mkdir -p "$OUTPUT_BASE_DIR"


find "$DATA_DIR" -name "*.json" -type f -printf '%h\n' | sort -u | while read dir_path; do

    relative_path="${dir_path#$DATA_DIR/}"
    
   
    output_dir="$OUTPUT_BASE_DIR/$relative_path"
    output_path="$output_dir"
    

    mkdir -p "$output_dir"
    
    echo "Processing directory: $dir_path"
    echo "Output will be saved to: $output_path"
    
    # Run the judge
    python llm_as_judge.py \
    --input_path "$dir_path" \
    --output_path "$output_path" \
    --model deepseek-chat \
    --api_base https://api.deepseek.com \
    --max_workers 10

    # 添加简单的错误检查
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $dir_path"
    else
        echo "✗ Failed to process: $dir_path"
    fi
    
    echo "---"
done

echo "All files processed!"

