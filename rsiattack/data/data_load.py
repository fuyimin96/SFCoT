import torchvision.transforms as transforms
import os
from torchvision.datasets import ImageFolder
import pandas as pd
import PIL
from torch.utils import data
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from ..model import func
model_box = func.models_mapping

label_MTARSI = {'A-10': 0, 'A-26': 1, 
                'B-1': 2, 'B-2': 3, 'B-29': 4, 
                'B-52': 5, 'Boeing': 6, 
                'C-130': 7, 'C-135': 8, 
                'C-17': 9, 'C-21': 10, 'C-5': 11, 'E-3': 12, 
                'F-16': 13, 'F-22': 14, 'KC-10': 15, 
                'P-63': 16, 'T-43': 17, 'T-6': 18, 'U-2': 19}

label_FGSCR_42 = {'001.Nimitz-class_aircraft_carrier': 0, '006.INS_Vikramaditya_aircraft_carrier': 1,
                '036.Kongo-class_destroyer': 2, '013.Type_45_destroyer': 3, '030.Civil_yacht': 4,
                '020.Freedom-class_combat_ship': 5, '007.INS_Virrat_aircraft_carrier': 6,
                '012.Kidd-class_destroyer': 7, '029.Towing_vessel': 8, '008.Ticonderoga-class_cruiser': 9,
                '017.Lzumo-class_helicopter_destroyer': 10, '028.Container_ship': 11,
                '014.Wasp-class_assault_ship': 12, '005.Charles_de_Gaulle_aricraft_carrier': 13,
                '015.Osumi-class_landing_ship': 14, '016.Hyuga-class_helicopter_destroyer': 15,
                '025.Megayacht': 16, '038.Atago-class_destroyer': 17, '011.Asagiri-class_destroyer': 18,
                '037.Horizon-class_destroyer': 19, '027.Murasame-class_destroyer': 20,
                '021.Independence-class_combat_ship': 21, '009.Arleigh_Burke-class_destroyer': 22, 
                '004.Kuznetsov-class_aircraft_carrier': 23, '031.Medical_ship': 24, 
                '041.Mistral-class_amphibious_assault_ship': 25, 
                '040.Juan_Carlos_I_Strategic_Projection_Ship': 26, 
                '034.Garibaldi_aircraft_carrier': 27, '023.Crane_ship': 28, 
                '019.San_Antonio-class_transport_dock': 29, 
                '042.San_Giorgio-class_transport_dock': 30, '010.Akizuki-class_destroyer': 31, 
                '018.Whitby_Island-class_dock_landing_ship': 32, '002.KittyHawk-class_aircraft_carrier': 33, 
                '033.Tank_ship': 34, '003.Midway-class_aircraft_carrier': 35, 
                '039.Maestrale-class_frigate': 36, 
                '026.Cargo_ship': 37, '035.Zumwalt-class_destroyer': 38,
                '032.Sand_carrier': 39, '022.Sacramento-class_support_ship': 40, 
                '024.Abukuma-class_frigate': 41}

label_FGSC23 = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
 '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, 
 '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, 
 '18': 18, '19': 19, '20': 20, '21': 21, '22': 22}


label_mapping = {'FGSCR_42': label_FGSCR_42,
                  'MTARSI': label_MTARSI,}


class customFolder(ImageFolder):  
    def __init__(self, root, transform=None):
        super(customFolder, self).__init__(root, transform)
        self.path_list = [path for path, target in self.samples]
        
    def __getitem__(self, index: int) :
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path
    
    def __len__(self):
        return super().__len__()


data_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

class trainDataset(data.Dataset):
    def __init__(self, model: str, args) -> None:
        """_summary_

        Args:
            model (_type_): 'train' or 'test'
            args (_type_): argparse
        """        
        self.data_dir = os.path.join(args.data_dir,
                                     args.data_type,
                                    model)
        self.data_path = []
        self.data_label = []
        self.label_box = label_mapping[args.data_type]
        for subdir in self.label_box.keys():
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                # 获取子文件夹中的文件名
                for file_path in glob.glob(os.path.join(subdir_path, '*')):
                    self.data_path.append(file_path)
                    self.data_label.append(subdir)
    
    def __getitem__(self, index):
        image_path = self.data_path[index]
        image = PIL.Image.open(image_path).convert('RGB')  
        image = data_transform(image)
        label = self.label_box[self.data_label[index]]
        return image, label
    
    def __len__(self):
        return len(self.data_path)


class attackDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_dir = os.path.join(args.data_dir,
                                     args.data_type,
                                    'test')
        csv_path = 'logs/{}_list.csv'.format(args.data_type)
        if not os.path.exists(csv_path):
            print('no the list of data, build new list')
            get_images_lists(args.data_dir, args.data_type, args.device)

        label_box = os.listdir(self.data_dir)
        self.label_dict = label_mapping[args.data_type]
        self.name_dict = {self.label_dict[string] : string for string in self.label_dict.keys()}

        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path, index_col=0)
            self.path_box = self.df.columns.to_list()
        else:
            raise Exception('no file name {}, please run \
                            the code of get_images_lists()'.format(csv_path))

    def __getitem__(self, index):
        image_path = os.path.join(self.args.data_dir, 
                                  self.args.data_type, 'test')  + self.path_box[index]
        image_path = image_path.replace('/', os.sep)
        image_path = image_path.replace('\\', os.sep)
        image = PIL.Image.open(image_path).convert('RGB')
        image = data_transform(image)

        (dir_name, file_name) = os.path.split(image_path)
        (_, label_name) = os.path.split(dir_name)

        te = torch.tensor(self.label_dict[label_name])
        
        return image, te, file_name

    def __len__(self):
        return len(self.df.columns)


class evalDataset(attackDataset):
    def __init__(self, adv_dir, args):
        super(evalDataset, self).__init__(args)
        self.adv_dir = adv_dir
    
    def __getitem__(self, index):
        image_path = os.path.join(self.args.data_dir, 
                                  self.args.data_type, 'test')  + self.path_box[index]
        image_path = image_path.replace('/', os.sep)
        image_path = image_path.replace('\\', os.sep)
        image_clean = PIL.Image.open(image_path).convert('RGB')  
        image_clean = data_transform(image_clean)

        (dir_name, file_name) = os.path.split(image_path)
        (_, label_name) = os.path.split(dir_name)

        adv_path = os.path.join(self.adv_dir, label_name,
                                file_name)

        image_adv = PIL.Image.open(adv_path).convert('RGB')  
        image_adv = data_transform(image_adv)

        return image_clean, image_adv


def get_images_lists(data_dir, data_type, device = 'cuda:0'):
    """ Obtain a list of images that are correctly recognized across all images.  """
    data_dir = os.path.join(data_dir, data_type, 'test')
    dataset = customFolder(data_dir, data_transform)
    loader = DataLoader(dataset, batch_size=32,
                        shuffle=False, num_workers=5)
    paths = dataset.path_list
    paths = [path.replace(data_dir, "")  for path in paths]
    paths = [path.replace("\\", "/")  for path in paths]
    df = pd.DataFrame(columns=paths)
    
    
    for key, fc in model_box.items():
        model_name = key + '_' + data_type + '.pt'
        model_path = os.path.join('./checkpoints', model_name)
        model_dict = torch.load(model_path, map_location='cpu')
        if data_type == 'FGSCR_42':
            num_classes = 42
        elif data_type == 'MTARSI': 
            num_classes = 20
        net = fc(num_classes)
        net.load_state_dict(model_dict)
        net = net.to(device)
        net.eval()
        for data, true_label, paths in loader:
            paths = [path.replace(data_dir, "")  for path in paths]
            paths = [path.replace("\\", "/")  for path in paths]
            df.loc[key, paths] = true_label.tolist()
        with torch.no_grad():
            for data, true_label, paths in tqdm(loader):
                paths = [path.replace(data_dir, "")  for path in paths]
                paths = [path.replace("\\", "/")  for path in paths]
                data, true_label = data.to(device), true_label.to(device)
                y_hat = F.softmax(net(data), dim=1)
                pred = torch.argmax(y_hat, 1)
                df.loc[key, paths] = pred.tolist()
            
    
    for image_path in df.columns:
        if not df[image_path].nunique() == 1:
            df = df.drop(columns=image_path)
    df.to_csv('logs/{}_list.csv'.format(data_type))