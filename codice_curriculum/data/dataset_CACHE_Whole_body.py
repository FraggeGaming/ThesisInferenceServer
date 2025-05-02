import csv
import os
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import apply_transform
import data.config

class MioDataset(Dataset):
    def __init__(self, opt, folder_paths):
        self.opt = opt
        self.folder_paths = folder_paths
        self.files_A, self.files_B = self._load_files()


    def _load_files(self):
        files_A = []
        files_B = []
        for file_path in self.folder_paths:
                file_path = file_path.strip()
                files_A.append(file_path)
                # If there's no paired file, you can do:
                files_B.append(file_path)  # or None, if your model allows it

        return files_A, files_B

    def __getitem__(self, index):
        img_path_A = self.files_A[index]
        img_path_B = self.files_B[index]
        files_dict = [{'A': img_path_A, 'B': img_path_B}]

        # Accedi all'elemento appropriato della lista risultante
        file_dict = files_dict[0]

        return {'A': file_dict['A'], 'B': file_dict['B'], 'A_paths': img_path_A, 'B_paths': img_path_B}

    def __len__(self):
        return len(self.files_B)

def CreateDataloader(opt, shuffle=True, cache=False, folder_paths = []):
    """folder_paths = []

    with open(f'{opt.dataroot}/{opt.phase}_{opt.district}.csv' if opt.phase=='train' else f'{opt.dataroot}/{opt.phase}.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        #next(csv_reader)
        for row in csv_reader:
            folder_paths.append(row[0])"""

    if not folder_paths:
        print(f"[INFO] Nessun percorso di cartella trovato nel file {folder_paths}")
        return None
    print(folder_paths)
    

    mio_dataset = MioDataset(opt, folder_paths)

    # Decide se utilizzare CacheDataset o Dataset in base al valore di 'cache'
    if cache:
        ds = CacheDataset(data=mio_dataset, transform=data.config.train_transforms if opt.phase=='train' else data.config.test_transforms, cache_rate=0.55)
    else:
        ds = Dataset(data=mio_dataset, transform=data.config.train_transforms if opt.phase=='train' else data.config.test_transforms)

    data_loader = DataLoader(ds, batch_size=opt.batchSize, shuffle=shuffle, pin_memory=True)

    return data_loader




