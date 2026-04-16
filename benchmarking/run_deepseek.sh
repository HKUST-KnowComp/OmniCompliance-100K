
DATA_DIR="../data/cases"
OUTPUT_BASE_DIR="./result_deepseek"

mkdir -p "$OUTPUT_BASE_DIR"


find "$DATA_DIR" -name "*.json" -type f -printf '%h\n' | sort -u | while read dir_path; do

    relative_path="${dir_path#$DATA_DIR/}"
    
    output_dir="$OUTPUT_BASE_DIR/$relative_path"
    output_path="$output_dir"

    mkdir -p "$output_dir"
    
    echo "Processing directory: $dir_path"
    echo "Output will be saved to: $output_path"

    python case_eval_llm.py \
        --input_file_path "$dir_path" \
        --output_file_path "$output_path" \
        --model_type api \
        --max_workers 10 \

    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed: $dir_path"
    else
        echo "✗ Failed to process: $dir_path"
    fi
    
    echo "---"
done

echo "All files processed!"
