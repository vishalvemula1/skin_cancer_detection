import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import random

# Configuration - just change these paths
METADATA_PATH = 'HAM10000_metadata.csv'
IMAGE_FOLDER_1 = 'HAM10000_images_part_1'
IMAGE_FOLDER_2 = 'HAM10000_images_part_2'
OUTPUT_DIR = 'new_data_balanced'
TARGET_SIZE = (300, 225)  # Width, Height for your model

# Classification mapping
MALIGNANT = ['akiec', 'bcc', 'mel']
BENIGN = ['bkl', 'df', 'nv', 'vasc']

def setup_folders():
    """Create output directories"""
    folders = ['train/benign', 'train/malignant', 'test/benign', 'test/malignant']
    for folder in folders:
        Path(OUTPUT_DIR, folder).mkdir(parents=True, exist_ok=True)

def find_image(image_id):
    """Find image in either part_1 or part_2 folder"""
    for folder in [IMAGE_FOLDER_1, IMAGE_FOLDER_2]:
        path = Path(folder, f"{image_id}.jpg")
        if path.exists():
            return path
    return None

def process_image(source_path, target_path, augment=False):
    """Resize image and optionally apply simple augmentation"""
    try:
        img = Image.open(source_path).convert('RGB')
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        
        # Simple augmentation for malignant class
        if augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Slight brightness adjustment
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.1))
        
        img.save(target_path, 'JPEG', quality=95)
        return True
    except:
        return False

def main():
    print("🔄 Starting HAM10000 preprocessing...")
    
    # Setup
    setup_folders()
    
    # Load data
    print("📊 Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    df['class'] = df['dx'].apply(lambda x: 'malignant' if x in MALIGNANT else 'benign')
    
    # Show distribution
    counts = df['class'].value_counts()
    print(f"Original: {counts['benign']} benign, {counts['malignant']} malignant")
    
    # Train/test split (stratified)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Process test images (no augmentation)
    print("🧪 Processing test images...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        source = find_image(row['image_id'])
        if source:
            target = Path(OUTPUT_DIR, 'test', row['class'], f"{row['image_id']}.jpg")
            process_image(source, target)
    
    # Process training images
    print("🎯 Processing training images...")
    
    # Count malignant vs benign in training set
    train_counts = train_df['class'].value_counts()
    malignant_count = train_counts['malignant']
    benign_count = train_counts['benign']
    target_malignant = min(benign_count, malignant_count * 3)  # Don't over-augment
    
    print(f"Will augment malignant from {malignant_count} to {target_malignant}")
    
    # Process benign training images (no augmentation needed)
    benign_train = train_df[train_df['class'] == 'benign']
    for _, row in tqdm(benign_train.iterrows(), total=len(benign_train), desc="Benign"):
        source = find_image(row['image_id'])
        if source:
            target = Path(OUTPUT_DIR, 'train', 'benign', f"{row['image_id']}.jpg")
            process_image(source, target)
    
    # Process malignant training images (with augmentation)
    malignant_train = train_df[train_df['class'] == 'malignant']
    augmentations_needed = target_malignant - malignant_count
    aug_per_image = max(1, augmentations_needed // malignant_count)
    
    augmented_count = 0
    for _, row in tqdm(malignant_train.iterrows(), total=len(malignant_train), desc="Malignant"):
        source = find_image(row['image_id'])
        if not source:
            continue
            
        # Save original
        target = Path(OUTPUT_DIR, 'train', 'malignant', f"{row['image_id']}.jpg")
        process_image(source, target)
        
        # Create augmented versions
        for i in range(aug_per_image):
            if augmented_count >= augmentations_needed:
                break
            aug_target = Path(OUTPUT_DIR, 'train', 'malignant', f"{row['image_id']}_aug_{i}.jpg")
            if process_image(source, aug_target, augment=True):
                augmented_count += 1
        
        if augmented_count >= augmentations_needed:
            break
    
    # Final count
    print("\n✅ Preprocessing complete!")
    for split in ['train', 'test']:
        for cls in ['benign', 'malignant']:
            count = len(list(Path(OUTPUT_DIR, split, cls).glob('*.jpg')))
            print(f"{split}/{cls}: {count} images")
    
    print(f"\n🎉 Dataset ready at '{OUTPUT_DIR}/' for your PyTorch model!")

if __name__ == "__main__":
    main()
