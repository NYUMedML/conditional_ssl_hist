import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset

class TCGA_CPTAC_Dataset(Dataset):
    def __init__(self, cptac_dir, tcga_dir, split_dir, transform=None, mode='train', batch_slide_num=4, batch_size=128):
        self.cptac_dir = cptac_dir
        self.tcga_dir = tcga_dir

        slide_list = pickle.load(open(split_dir + '/case_split.pkl', 'rb'))[mode + "_id"]
        # slide_list = [s for s in slide_list if "TCGA" in s]
        self.slide2tiles = {}
        for slide_id in slide_list:
            if "TCGA" in slide_id:
                self.slide2tiles[slide_id] = os.listdir(os.path.join(self.tcga_dir, slide_id))
            else:
                self.slide2tiles[slide_id] = os.listdir(os.path.join(self.cptac_dir, slide_id))
        self.idx2tiles = [os.path.join(slide_id, tile_name) for slide_id, tile_list in self.slide2tiles.items()
                          for tile_name in tile_list if 'jpg' in tile_name]
        self.tiles2idx = dict(zip(self.idx2tiles, range(len(self.idx2tiles))))
        self.idx2slide = slide_list
        self.slide2idx = dict(zip(self.idx2slide, range(len(self.idx2slide))))
        self.transform = transform
        self.batch_slide_num = batch_slide_num
        self.batch_size = batch_size

    def __getitem__(self, index):
        tile_names = []
        slide_id = self.idx2slide[index]
        selected_tiles = [slide_id + '/' + t for t in np.random.choice(self.slide2tiles[slide_id], self.batch_size // self.batch_slide_num)]
        tile_names += selected_tiles
        for i in range(self.batch_slide_num - 1):
            slide_id = self.idx2slide[np.random.randint(len(self.idx2slide))]
            tile_names += [slide_id + '/' + t for t in np.random.choice(self.slide2tiles[slide_id], self.batch_size // self.batch_slide_num)]
        indices = []
        imgs = []
        for tile_name in tile_names:
            if "TCGA" in tile_name:
                image = cv2.imread(self.tcga_dir + tile_name)
            else:
                image = cv2.imread(self.cptac_dir + tile_name)
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
            imgs.append(image_tensor)
            indices.append(index)
        return imgs, indices

    def __len__(self):
        return len(self.idx2slide)


class TCGA_CPTAC_Bag_Dataset(Dataset):
    def __init__(self, data_dir, split_dir, mode='train'):
        self.data_dir = data_dir
        slide_list = pickle.load(open(os.path.join(split_dir, 'case_split_2yr.pkl'), 'rb'))[mode + '_id']
        self.slide2tiles = {}
        for slide_id in slide_list:
            if "TCGA" in slide_id:
                tile_dir = self.data_dir + '/TCGA/tiles/'
            else:
                tile_dir = self.data_dir + '/CPTAC/tiles/'
            self.slide2tiles[slide_id] = os.listdir(os.path.join(tile_dir, slide_id))
        self.idx2tiles = [os.path.join(slide_id, tile_name) for slide_id, tile_list in self.slide2tiles.items() 
                     for tile_name in tile_list if 'jpg' in tile_name]
        self.tiles2idx = dict(zip(self.idx2tiles, range(len(self.idx2tiles))))
        self.idx2slide = slide_list
        self.slide2idx = dict(zip(self.idx2slide, range(len(self.idx2slide))))
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        tile_path = self.idx2tiles[index]
        slide_id, tile_name = tile_path.split('/')
        tile_idx = self.tiles2idx[tile_path]
        slide_idx = self.slide2idx[slide_id]
        if "TCGA" in tile_path:
            prefix = self.data_dir + '/TCGA/tiles/'
        else:
            prefix = self.data_dir + '/CPTAC/tiles/'
        image = cv2.imread(os.path.join(prefix, tile_path))
        image = Image.fromarray(image)
        image_tensor = self.transform(image)
        return image_tensor, tile_idx, slide_idx
    
    def __len__(self):
        return len(self.idx2tiles)



