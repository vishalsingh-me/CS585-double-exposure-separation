import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision.transforms import functional as TF
from pathlib import Path
import math

class SyntheticDoubleExposureDataset(Dataset):
    def __init__(self, manifest_csv, image_root, split="train", size=256, 
                 mode="linear_clean", same_label_probability=0.1, transform=None):
        self.manifest_csv = Path(manifest_csv)
        self.image_root = Path(image_root)
        self.split = split
        self.size = size
        self.mode = mode
        self.same_label_probability = same_label_probability
        self.transform = transform
        
        with open(self.manifest_csv, 'r') as f:
            all_rows = list(csv.DictReader(f))
            
        self.rows = [r for r in all_rows if r.get("split", self.split) == self.split]
        
        self.label_dict = {}
        for r in self.rows:
            lbl = r.get("label", "unknown")
            if lbl not in self.label_dict:
                self.label_dict[lbl] = []
            self.label_dict[lbl].append(r)
            
    def __len__(self):
        return len(self.rows)
        
    def _sample_second(self, first_row):
        same_label = False
        lbl = first_row.get("label", "unknown")
        
        if random.random() < self.same_label_probability and len(self.label_dict.get(lbl, [])) > 1:
            candidates = [r for r in self.label_dict[lbl] if r != first_row]
            if candidates:
                return random.choice(candidates), True
                
        candidates = [r for r in self.rows if r != first_row]
        return random.choice(candidates), False

    def __getitem__(self, idx):
        row1 = self.rows[idx]
        row2, same_label = self._sample_second(row1)
        
        p1 = self.image_root / row1.get("relative_path", row1.get("path", ""))
        p2 = self.image_root / row2.get("relative_path", row2.get("path", ""))
        
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        
        # simple resize crop
        img1 = TF.resize(img1, [self.size, self.size])
        img2 = TF.resize(img2, [self.size, self.size])
        
        t1 = TF.to_tensor(img1)
        t2 = TF.to_tensor(img2)
        
        alpha = random.uniform(0.3, 0.7)
        
        if self.mode == "linear_clean":
            mix = alpha * t1 + (1 - alpha) * t2
        elif self.mode == "mask_clean":
            mask = torch.zeros(1, self.size, self.size)
            for y in range(self.size):
                for x in range(self.size):
                    dist = math.sqrt((x - self.size/2)**2 + (y - self.size/2)**2)
                    val = max(0, 1 - dist / (self.size/2))
                    mask[0, y, x] = val
            mix = mask * t1 + (1 - mask) * t2
        else:
            mix = alpha * t1 + (1 - alpha) * t2
            
        if self.transform:
            mix = self.transform(mix)
            
        metadata = {
            "source_1_path": str(p1),
            "source_2_path": str(p2),
            "split": self.split,
            "synthesis_mode": self.mode,
            "same_label": same_label,
            "alpha": alpha
        }
        
        return {
            "mixture": mix,
            "source_1": t1,
            "source_2": t2,
            "metadata": metadata
        }

def custom_collate(batch):
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
