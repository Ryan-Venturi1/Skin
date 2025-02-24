import os
import requests
import zipfile
import io
from tqdm import tqdm
import glob

# Configuration
OUTPUT_DIR = 'isic_dataset'
TEMP_DIR = 'temp_downloads'
chunk_size = 1024 * 1024
# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Direct download URLs for ISIC datasets
DATASETS = [
    {
        "name": "ISIC_2018_Task3_Training",
        "image_url": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip",
        "metadata_url": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip",
        "classes": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    }
]

def download_file(url, output_path):
    """Download a file from URL to the specified path with progress bar"""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path
        
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(url),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            bar.update(size)
    
    return output_path

def extract_zip(zip_path, extract_to):
    """Extract a ZIP file to the specified directory"""
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_dataset():
    """Download and process the ISIC 2018 dataset"""
    # For simplicity, we'll just use the first dataset in our list
    dataset = DATASETS[0]
    
    # Download image ZIP
    image_zip_path = os.path.join(TEMP_DIR, f"{dataset['name']}_images.zip")
    download_file(dataset['image_url'], image_zip_path)
    
    # Download metadata ZIP
    metadata_zip_path = os.path.join(TEMP_DIR, f"{dataset['name']}_metadata.zip")
    download_file(dataset['metadata_url'], metadata_zip_path)
    
    # Extract image ZIP
    image_extract_path = os.path.join(TEMP_DIR, "images")
    os.makedirs(image_extract_path, exist_ok=True)
    extract_zip(image_zip_path, image_extract_path)
    
    # Extract metadata ZIP
    metadata_extract_path = os.path.join(TEMP_DIR, "metadata")
    os.makedirs(metadata_extract_path, exist_ok=True)
    extract_zip(metadata_zip_path, metadata_extract_path)
    
    # Find the ground truth CSV file
    csv_pattern = os.path.join(metadata_extract_path, '**', '*.csv')
    csv_files = glob.glob(csv_pattern, recursive=True)
    if not csv_files:
        print(f"No CSV files found in {metadata_extract_path}")
        print("Contents of the directory:")
        print(os.listdir(metadata_extract_path))
        raise FileNotFoundError("Ground truth CSV file not found")

    ground_truth_path = csv_files[0]
    print(f"Found ground truth file: {ground_truth_path}")
    
    # Process ground truth
    print("Organizing images by diagnosis...")
    import csv
    
    # Map from abbreviated to full class names
    class_mapping = {
        "MEL": "melanoma",
        "NV": "nevus",
        "BCC": "basal_cell_carcinoma",
        "AKIEC": "actinic_keratosis",
        "BKL": "benign_keratosis",
        "DF": "dermatofibroma",
        "VASC": "vascular_lesion"
    }
    
    # Create class directories
    class_dirs = {}
    for abbrev, full_name in class_mapping.items():
        class_dir = os.path.join(OUTPUT_DIR, full_name)
        os.makedirs(class_dir, exist_ok=True)
        class_dirs[abbrev] = class_dir
    
    # Process and organize images
    with open(ground_truth_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Organizing images"):
            # Determine the class with highest probability
            best_class = None
            best_prob = 0
            
            for abbrev in class_mapping:
                if abbrev in row and float(row[abbrev]) > best_prob:
                    best_prob = float(row[abbrev])
                    best_class = abbrev
            
            if best_class:
                image_id = row['image']
                source_path = os.path.join(image_extract_path, "ISIC2018_Task3_Training_input", f"{image_id}.jpg")
                if os.path.exists(source_path):
                    target_dir = class_dirs[best_class]
                    target_path = os.path.join(target_dir, f"{image_id}.jpg")
                    
                    # Copy the image file
                    import shutil
                    shutil.copy2(source_path, target_path)
    
    # Report results
    class_counts = {}
    for cls in class_mapping.values():
        class_dir = os.path.join(OUTPUT_DIR, cls)
        count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
        class_counts[cls] = count
    
    print("\nDataset successfully organized:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} images")

if __name__ == "__main__":
    process_dataset()
    print("\nDataset preparation complete!")