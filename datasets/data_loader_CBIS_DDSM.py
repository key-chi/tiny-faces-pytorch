#import config as cfg
import pydicom
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image

class CBISDDSMDataset(Dataset):

    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels, self.images = self.read_labels(label_file)

    def read_labels(self, label_file):
        df = pd.read_csv(label_file)
        
        #for column in df.columns:
        #    unique_values = df[column].unique()
        #    print(f"Unique values in '{column}': {unique_values}")

        # Combine 'BENIGN' and 'BENIGN_WITHOUT_CALLBACK' into 'BENIGN'
        df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')

        # Initialize lists for images, pathologies, and ratios
        images = []
        labels = []

        for _, row in df.iterrows():
            # Pathology label
            pathology = row['pathology']
            ratio = 0

            # Load ROI mask image
            roi_mask_dir = f'{self.image_dir}/{row["ROI mask file path"]}'
            roi_mask_filename = '1-2.dcm' # we know that the mask is always 1-2.dcm
            roi_mask_filepath = self.find_file_in_directory(roi_mask_dir, roi_mask_filename, 'ROI mask')

            if roi_mask_filepath is not None:
                roi_mask = pydicom.dcmread(roi_mask_filepath)
                roi_mask_np = np.array(roi_mask.pixel_array)

                # Calculate white to black pixel ratio
                white_pixels = np.sum(roi_mask_np == 255)
                black_pixels = np.sum(roi_mask_np == 0)
                ratio = white_pixels / black_pixels if black_pixels != 0 else 0
            else:
                print(f"ROI mask file '{roi_mask_filename}' not found in directory '{roi_mask_dir}'")

            # Append to lists
            image_dir =  f'{self.image_dir}/{row["image file path"]}'
            img_filename = '1-1.dcm' # we know that the mask is always 1-1.dcm
            img_filepath = self.find_file_in_directory(image_dir, img_filename, 'full mammogram')

            if img_filepath is None:
                print(f"Image file '{img_filename}' not found in directory '{image_dir}'")

            images.append(img_filepath)
            labels.append({'pathology': pathology, 'ratio': ratio})

        return labels, images
    
    def find_file_in_directory(self, file_path, target_file_name, target_subdir_fallback):
        # Extract the base directory from the file path
        #print(f'file_path={file_path}')
        base_directory = file_path.split('/Mass', 1)[0]
        #print(f'base_directory={base_directory}')

        # Search for the target file in the base directory
        for root, dirs, files in os.walk(base_directory):
            if target_file_name in files:
                return os.path.join(root, target_file_name)
            
        # Secondary search in subdirectories containing target_subdir_fallback name
        # because sometimes the dataset is crap and didn't follow the naming convention... >:(
        for root, dirs, files in os.walk(base_directory):
            if target_subdir_fallback in root and files:
                # Assuming there's only one file in the directory
                return os.path.join(root, files[0])

        return None  # Return None if the file is not found

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        dicom_data = pydicom.dcmread(img_path)
        image = dicom_data.pixel_array
        image = Image.fromarray(image).convert('RGB')  # Convert to RGB

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        class_float = 0.0 if label['pathology'] == 'BENIGN' else 1.0
        class_tensor = torch.tensor(class_float, dtype=torch.float)

        ratio_float = label['ratio']
        ratio_tensor = torch.tensor(ratio_float, dtype=torch.float)

        return image, class_tensor, ratio_tensor


def load_data(cbis_ddsm_path, cbis_ddsm_train_labels, val_ratio, batch_size, img_transforms):
    '''_transform = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.ToTensor()
    ])'''

    train_dataset = CBISDDSMDataset(image_dir=cbis_ddsm_path, label_file=cbis_ddsm_train_labels, transform=img_transforms)
    #test_dataset = CBISDDSMDataset(image_dir=cbis_ddsm_path, label_file=cbis_ddsm_test_labels, transform=_transform)

    # Split dataset into train, validation, and test sets
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    #print(f"Test size: {len(test_loader.dataset)}")

    return train_loader, val_loader#, test_loader


# You can add additional functions or classes if needed
if __name__ == '__main__':
    load_data()