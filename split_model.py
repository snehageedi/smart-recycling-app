import os
import math

INPUT_FILE = 'smart_waste_classifier_model.h5'
MAX_CHUNK_SIZE_MB = 20 # Keep it well under 25MB for safety
MAX_CHUNK_SIZE_BYTES = MAX_CHUNK_SIZE_MB * 1024 * 1024
OUTPUT_PREFIX = 'model_part_'

file_size = os.path.getsize(INPUT_FILE)
num_chunks = math.ceil(file_size / MAX_CHUNK_SIZE_BYTES)

print(f"File size: {file_size / (1024*1024):.2f} MB. Splitting into {num_chunks} parts.")

with open(INPUT_FILE, 'rb') as f:
    for i in range(num_chunks):
        chunk_data = f.read(MAX_CHUNK_SIZE_BYTES)
        chunk_filename = f'{OUTPUT_PREFIX}{i:02d}' # e.g., model_part_00, model_part_01
        
        with open(chunk_filename, 'wb') as chunk_file:
            chunk_file.write(chunk_data)
        
        print(f"Created {chunk_filename} ({os.path.getsize(chunk_filename) / (1024*1024):.2f} MB)")

print("Splitting complete.")