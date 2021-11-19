import random
import os
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
import torch.utils.data as data
import fnmatch

def resize_instrinsic(intrinsic, scale_x, scale_y):
  intrinsic_rsz = np.copy(intrinsic)
  intrinsic_rsz[0, :] *= scale_x
  intrinsic_rsz[1, :] *= scale_y
  return intrinsic_rsz


def raw_city_sequences():
  """Sequence names for city sequences in kitti raw data.
  Returns:
    seq_names: list of names
  """
  seq_names = [
      '2011_09_26_drive_0001',
      '2011_09_26_drive_0002',
      '2011_09_26_drive_0005',
      '2011_09_26_drive_0009',
      '2011_09_26_drive_0011',
      '2011_09_26_drive_0013',
      '2011_09_26_drive_0014',
      '2011_09_26_drive_0017',
      '2011_09_26_drive_0018',
      '2011_09_26_drive_0048',
      '2011_09_26_drive_0051',
      '2011_09_26_drive_0056',
      '2011_09_26_drive_0057',
      '2011_09_26_drive_0059',
      '2011_09_26_drive_0060',
      '2011_09_26_drive_0084',
      '2011_09_26_drive_0091',
      '2011_09_26_drive_0093',
      '2011_09_26_drive_0095',
      '2011_09_26_drive_0096',
      '2011_09_26_drive_0104',
      '2011_09_26_drive_0106',
      '2011_09_26_drive_0113',
      '2011_09_26_drive_0117',
      '2011_09_28_drive_0001',
      '2011_09_28_drive_0002',
      '2011_09_29_drive_0026',
      '2011_09_29_drive_0071',
  ]
  return seq_names


def _collate_fn(batch):
    _src_items, _tgt_items = zip(*batch)

    # Gather and stack tgt infos
    tgt_items = defaultdict(list)
    for si in _tgt_items:
        for k, v in si.items():
            tgt_items[k].append(default_collate(v))

    for k in tgt_items.keys():
        tgt_items[k] = torch.stack(tgt_items[k], axis=0)

    src_items = default_collate(_src_items)
    src_items = {k: v for k, v in src_items.items()
                 if k != "G_cam_world"}
    return src_items, tgt_items


class NeRFDataset(data.Dataset):
    def __init__(self, config, logger, root, is_validation, img_size,
                 ):
        self.logger = logger
        self.config = config
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.is_validation = is_validation
        self.collate_fn = _collate_fn
        self._init_img_transforms()

        self.root_dir  = root

        self.cam_calibration = defaultdict()
        self.init_img_names_seq_list()
        self.preload_calib_files()
       
        self.length = len(self.img_list_src)
        if self.logger:
            self.logger.info("Dataset root: {}, is_validation: {}, number of images: {}"
                             .format(root, self.is_validation, self.length))
    def __len__(self):
        return self.length

    def get_data(self, img_src,src_num, img_trg,trg_num, src_shape, trg_shape,
                       calib_data):
        """Single pair loader.
        Args:
        img_src: source image
        img_trg: trg image
        src_shape: source image original shape
        trg_shape: target image original shape
        calib_data: calibratio data for current sequence
        Returns:
        img_s: Source frame image
        img_t: Target frame image
        k_s: Source frame intrinsic
        k_t: Target frame intrinsic
        rot: relative rotation from source to target
        trans: relative translation from source to target
        """
        rot = np.eye(3)
        trans = np.zeros(3)
        k_s = np.copy(calib_data[f'P_rect_0{str(src_num)}'].reshape(3, 4)[:3, :3])

        k_t = np.copy(calib_data[f'P_rect_0{str(trg_num)}'].reshape(3, 4)[:3, :3])

        trans_src = np.copy(calib_data[f'P_rect_0{str(src_num)}'].reshape(3, 4)[:, 3])
        trans_trg = np.copy(calib_data[f'P_rect_0{str(trg_num)}'].reshape(3, 4)[:, 3])

        # The translation is in homogeneous 2D coords, convert to regular 3d space:
        trans_src[0] = (trans_src[0] - k_s[0, 2] * trans_src[2]) / k_s[0, 0]
        trans_src[1] = (trans_src[1] - k_s[1, 2] * trans_src[2]) / k_s[1, 1]

        trans_trg[0] = (trans_trg[0] - k_t[0, 2] * trans_trg[2]) / k_t[0, 0]
        trans_trg[1] = (trans_trg[1] - k_t[1, 2] * trans_trg[2]) / k_t[1, 1]

        trans = trans_trg - trans_src
        
        k_s = resize_instrinsic(k_s, self.img_w / src_shape[1], self.img_h / src_shape[0])
        k_t = resize_instrinsic(k_t, self.img_w / trg_shape[1], self.img_h / trg_shape[0])

        return (img_src, img_trg, k_s, k_t, rot, trans_trg.reshape(3, 1),trans_src.reshape(3, 1))
    def __getitem__(self, index):
        # Read src item
        img_path_src = self.img_list_src[index]
        img_path_trgt = self.img_list_trg[index]
        calib_data = self.cam_calibration[self.seq_id_list[index]]
        img_src = Image.open(img_path_src)
        img_tgt = Image.open(img_path_trgt)
        img_src_size = img_src.size #(w,h)
        # print("BEFORE RESIZE",img_src_size)
        img_tgt_size = img_tgt.size
        # if(self.is_validation):
        #     left_src = (img_src_size[0] - int(img_src_size[0]*0.9))/2
        #     top_src = (height - new_height)/2
        #     right_src = (width + new_width)/2
        #     bottom_src = (height + new_height)/2
        #     # Crop the center of the image
        #     im = im.crop((left, top, right, bottom))
        #             # print(img_src)
        # img_src_new = transforms.CenterCrop((int(img_src_size[1]*0.9),int(img_src_size[0]*0.9)))(img_src)
        # img_src_new.save(f"test_{index}.png")
        img_src = self.img_transforms(img_src)
        # print("AFTER RESIZE",img_src.size())
        img_tgt = self.img_transforms(img_tgt)
        src_num =2
        tgt_num=3
        if("image_03" in img_path_src and "image_02" in img_path_trgt):
            src_num = 3
            tgt_num =2
        img_src, img_trg, k_s, k_t, rot, trans_trg,trans_src = self.get_data(img_src,src_num,img_tgt,tgt_num,(img_src_size[1],img_src_size[0]),(img_tgt_size[1],img_tgt_size[0]),calib_data)
        # G_src = np.vstack([
        #     np.hstack((rot, trans_src)),
        #     np.array([0, 0, 0, 1])
        # ]).astype(np.float32)
        # G_trg = np.vstack([
        #     np.hstack((rot, trans_trg)),
        #     np.array([0, 0, 0, 1])
        # ]).astype(np.float32)
        # G_src_trg = G_src.copy()
        G_src_trg = np.vstack([
            np.hstack((rot, -trans_trg +trans_src)),
            np.array([0, 0, 0, 1])
        ]).astype(np.float32)
        # G_src_trg = G_src @ np.linalg.inv(G_trg)
        src_item = {
            "img":img_src,
            "K":k_s,
            "K_inv":np.linalg.inv(k_s),
        }
        tgt_item = {
            "img":img_trg,
            "K":k_t,
            "K_inv":np.linalg.inv(k_t),
            "G_src_tgt":G_src_trg
        }
        # print(src_item)
        return src_item, tgt_item

    def __len__(self):
        return self.length

    def _init_img_transforms(self):
        if(not self.is_validation):
            self.img_transforms = transforms.Compose([
                
                transforms.Resize((self.img_h, self.img_w), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.Resize((self.img_h, self.img_w), interpolation=Image.BICUBIC),
                transforms.CenterCrop((int(self.img_h*0.90),int(self.img_w*0.90))), #Crop 5% from each side
                transforms.Resize((self.img_h, self.img_w), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
        

    
    def init_img_names_seq_list(self):
        # Image list.
       
        self.img_list_src = []
        self.img_list_trg = []
        self.seq_id_list = []
        exclude_img = '2011_09_26_drive_0117_sync/image_02/data/0000000074.png'
        seq_names = raw_city_sequences()
        rng = np.random.RandomState(0)
        rng.shuffle(seq_names)
        n_all = len(seq_names)
        n_train = int(round(0.7 * n_all))
        n_val = int(round(0.15 * n_all))
        if not self.is_validation:
            seq_names = seq_names[0:n_train]
        else:
            seq_names = seq_names[(n_train + n_val):n_all]       
        for seq_id in seq_names:
            seq_date = seq_id[0:10]
            seq_dir = os.path.join(self.root_dir, seq_date,
                                '{}_sync'.format(seq_id))
            for root, _, filenames in os.walk(os.path.join(seq_dir, 'image_02')):
                for filename in fnmatch.filter(filenames, '*.png'):
                    src_img_name = os.path.join(root, filename)
                    if exclude_img not in src_img_name:
                        self.img_list_src.append(os.path.join(src_img_name))
                        self.seq_id_list.append(seq_date)
        self.img_list_trg = [
          f.replace('image_02', 'image_03') for f in self.img_list_src
         ]
        if(not self.is_validation):
            for i in range(len(self.img_list_src)):
                num= random.randint(0,1)
                if(num==1):
                    self.img_list_src[i] = self.img_list_src[i].replace('image_02', 'image_03')
                    self.img_list_trg[i] = self.img_list_trg[i].replace('image_03', 'image_02')
    def preload_calib_files(self):
        """Preload calibration files for the sequence."""
       
        seq_names = raw_city_sequences()
        for seq_id in seq_names:
            seq_date = seq_id[0:10]
            calib_file = os.path.join(self.root_dir, seq_date,
                                    'calib_cam_to_cam.txt')
            self.cam_calibration[seq_date] = self.read_calib_file(calib_file)

    def read_calib_file(self, file_path):
        """Read camera intrinsics."""
        # Taken from https://github.com/hunse/kitti
        float_chars = set('0123456789.e+- ')
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass

        return data    
   


if __name__ == "__main__":
    import logging
    dataset = NeRFDataset({}, logging,
                          root="/home/yafathi/scratch/kitti_raw_data",
                          is_validation=True,
                          img_size=(384, 256))
                         
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=10, shuffle=True,
                    drop_last=True, num_workers=0)
                    # collate_fn=_collate_fn)

    for batch in dl:
        src_item, supervision_items = batch

        for k, v in src_item.items():
            print(k, v.size())

        print("********")

        for k, v in supervision_items.items():
            print(k, v.size())

        break
