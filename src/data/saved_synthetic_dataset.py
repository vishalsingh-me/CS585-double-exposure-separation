import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SavedSyntheticDataset(Dataset):
    def __init__(self, manifest_csv, image_size=None):
        self.manifest_path = Path(manifest_csv)
        self.base_dir = self.manifest_path.parent
        self.image_size = image_size
        
        with open(self.manifest_path, 'r') as f:
            self.rows = list(csv.DictReader(f))
            
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        
        p_mix = self.base_dir / row["mixture_path"]
        p_s1 = self.base_dir / row["target_1_path"]
        p_s2 = self.base_dir / row["target_2_path"]
        
        mix_img = Image.open(p_mix).convert('RGB')
        s1_img = Image.open(p_s1).convert('RGB')
        s2_img = Image.open(p_s2).convert('RGB')
        
        t_mix = TF.to_tensor(mix_img)
        t_s1 = TF.to_tensor(s1_img)
        t_s2 = TF.to_tensor(s2_img)
        
        if self.image_size is not None:
            t_mix = TF.resize(t_mix, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            t_s1 = TF.resize(t_s1, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            t_s2 = TF.resize(t_s2, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            
        return {
            "mixture": t_mix,
            "source_1": t_s1,
            "source_2": t_s2,
            "metadata": row
        }

def custom_collate_saved(batch):
    mixtures = torch.stack([item['mixture'] for item in batch])
    source_1 = torch.stack([item['source_1'] for item in batch])
    source_2 = torch.stack([item['source_2'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        "mixture": mixtures,
        "source_1": source_1,
        "source_2": source_2,
        "metadata": metadata
    }
