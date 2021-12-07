import random
import os
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import glob
import imageio
import cv2

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)

def resize_instrinsic(intrinsic, scale_x, scale_y):
    intrinsic_rsz = np.copy(intrinsic)
    intrinsic_rsz[0, :] *= scale_x
    intrinsic_rsz[1, :] *= scale_y
    return intrinsic_rsz
def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )

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
                 supervision_count=1, visible_points_count=8, img_pre_downsample_ratio=7.875):
        self.logger = logger
        self.config = config
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.is_validation = is_validation
        self.visible_points_count = visible_points_count
        self.supervision_count = supervision_count
        self.collate_fn = _collate_fn
        self.scene_objs_to_indices = defaultdict(dict)
        self._init_img_transforms()
        
        self.base_path = root
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(self.base_path, "*")) if os.path.isdir(x)]
        list_prefix = "softras_"
        if not is_validation:
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        else:
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        all_objs = []
        all_imgs = []
        index  = 0
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            objs = []
            with open(file_list, "r") as f:
                for object in f.readlines():
                    self.scene_objs_to_indices[cat][object] = list()
                    img_list = glob.glob(os.path.join(os.path.join(base_dir, object.strip()),"image", "*"))
                    all_img_paths = list()
                    for image in img_list:
                       
                        if (image.endswith(".jpg") or image.endswith(".png")):
                            # print(image)
                            all_img_paths.append(image)
                    if(len(all_img_paths)==0):
                        continue
                    src_image = random.sample(all_img_paths,1)[0]
                    all_img_paths.remove(src_image)
                    src_all = [src_image] * len(all_img_paths)
                    for src,tgt in zip(src_all,all_img_paths):
                        all_imgs.append((cat,object, os.path.join(base_dir, object.strip()),os.path.join(base_dir, object.strip(),"image",os.path.basename(src)),os.path.join(base_dir, object.strip(),"image",os.path.basename(tgt))))

                    # for image in glob.glob(os.path.join(os.path.join(base_dir, object.strip()),"image", "*")):
                        
                    #     if (image.endswith(".jpg") or image.endswith(".png")):


                    #         self.scene_objs_to_indices[cat][object].append(index)
                    #         all_imgs.append((cat,object, os.path.join(base_dir, object.strip()),os.path.join(base_dir, object.strip(),"image",os.path.basename(image))))
                    #         index+=1
                    objs.append((cat,object, os.path.join(base_dir, object.strip())))
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.all_imgs = all_imgs
        self.stage = "train"

        self.image_to_tensor = get_image_to_tensor_balanced(img_size[0])
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading DVR dataset",
            self.base_path,
            "test",
            is_validation,
            len(self.all_imgs),
            "objs",
            
        )

        self.image_size = (128,128)

     
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self.sub_format = "shapenet"
        self.scale_focal = True
        self.length = len(all_imgs)
        # self.z_near = z_near
        # self.z_far = z_far
        # self.lindisp = False
        if self.logger:
            self.logger.info("Dataset root: {}, is_validation: {}, number of images: {}"
                             .format(root, self.is_validation, self.length))

    def __getitem__(self, index):
        # Read src item
        # print(index)
        cat,obj_name, root_dir,img_dir,tgt_img = self.all_imgs[index]
        # cat,obj_name,_ = self.all_objs[index]
        img_indices = self.scene_objs_to_indices[cat][obj_name]
        # print(img_indices)
        # src_img_idx = random.sample(img_indices,1)[0]
        # cat,obj_name, root_dir,img_dir = self.all_imgs[src_img_idx]
        # print(cat)
        i = int(os.path.basename(img_dir).split(".")[0])
        cam_path = os.path.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)
       

        focal=None
       
        img = Image.open(img_dir)
        # img.save(f"src_{index}.jpg")
        x_scale = img.size[0] / 2.0
        y_scale = img.size[1] / 2.0
        xy_delta = 1.0
        
        # ShapeNet
        wmat_inv_key = "world_mat_inv_" + str(i)
        wmat_key = "world_mat_" + str(i)
        if wmat_inv_key in all_cam:
            extr_inv_mtx = all_cam[wmat_inv_key]
        else:
            extr_inv_mtx = all_cam[wmat_key]
            if extr_inv_mtx.shape[0] == 3:
                extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
            extr_inv_mtx = np.linalg.inv(extr_inv_mtx)
        extr_mtx =  all_cam[wmat_key]
        intr_mtx = all_cam["camera_mat_" + str(i)][:3,:3]
        intr_mtx[0,0] *= x_scale
        intr_mtx[1,1] *= y_scale
        intr_mtx = resize_instrinsic(intr_mtx,2,2)
        intr_mtx_inv = np.linalg.inv(intr_mtx)[:3,:3]
        # fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
        # assert abs(fx - fy) < 1e-9
        # fx = fx * x_scale
        # if focal is None:
        #     focal = fx
        # else:
        #     assert abs(fx - focal) < 1e-5
        pose = extr_inv_mtx

        pose = (
            self._coord_trans_world
            @ torch.tensor(pose, dtype=torch.float32)
            @ self._coord_trans_cam
        )
        G_src_world = pose
        img_tensor = self.image_to_tensor(img)
        # if mask_path is not None:
        #     mask_tensor = self.mask_to_tensor(mask)

        #     rows = np.any(mask, axis=1)
        #     cols = np.any(mask, axis=0)
        #     rnz = np.where(rows)[0]
        #     cnz = np.where(cols)[0]
        #     if len(rnz) == 0:
        #         raise RuntimeError(
        #             "ERROR: Bad image at", rgb_path, "please investigate!"
        #         )
        #     rmin, rmax = rnz[[0, -1]]
        #     cmin, cmax = cnz[[0, -1]]
        #     bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
        #     all_masks.append(mask_tensor)
        #     all_bboxes.append(bbox)

        # all_imgs.append(img_tensor)
        # all_poses.append(pose)

        # if self.sub_format != "shapenet":
        #     fx /= len(rgb_paths)
        #     fy /= len(rgb_paths)
        #     cx /= len(rgb_paths)
        #     cy /= len(rgb_paths)
        #     focal = torch.tensor((fx, fy), dtype=torch.float32)
        #     c = torch.tensor((cx, cy), dtype=torch.float32)
        #     all_bboxes = None
        # elif mask_path is not None:
        #     all_bboxes = torch.stack(all_bboxes)

        # all_imgs = torch.stack(all_imgs)
        # all_poses = torch.stack(all_poses)
        # if len(all_masks) > 0:
        #     all_masks = torch.stack(all_masks)
        # else:
        #     all_masks = None

        # if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
        #     scale = self.image_size[0] / all_imgs.shape[-2]
        #     focal *= scale
        #     if self.sub_format != "shapenet":
        #         c *= scale
        #     elif mask_path is not None:
        #         all_bboxes *= scale

        #     all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
        #     if all_masks is not None:
        #         all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")


        

        # Copy new src_item
        src_item = {
            "img":img_tensor,
            "K":intr_mtx,
            "K_inv":intr_mtx_inv,
        }
        tgt_i = int(os.path.basename(tgt_img).split(".")[0])
        cam_path = os.path.join(root_dir, "cameras.npz")
        all_cam = np.load(cam_path)
       

        focal=None
        tgt_image = Image.open(tgt_img)
        # tgt_image.save(f"src_{int(tgt_i)}_{index}.jpg")
        
        
        # ShapeNet
        wmat_inv_key = "world_mat_inv_" + str(tgt_i)
        wmat_key = "world_mat_" + str(tgt_i)
        if wmat_inv_key in all_cam:
            tgt_extr_inv_mtx = all_cam[wmat_inv_key]
        else:
            tgt_extr_inv_mtx = all_cam[wmat_key]
            if tgt_extr_inv_mtx.shape[0] == 3:
                tgt_extr_inv_mtx = np.vstack((tgt_extr_inv_mtx, np.array([0, 0, 0, 1])))
            tgt_extr_inv_mtx = np.linalg.inv(tgt_extr_inv_mtx)
        tgt_extr_mtx=  all_cam[wmat_key]
        tgt_intr_mtx = all_cam["camera_mat_" + str(tgt_i)][:3,:3]
        tgt_intr_mtx[0,0] *= x_scale
        tgt_intr_mtx[1,1] *= y_scale
        tgt_intr_mtx = resize_instrinsic(tgt_intr_mtx,2,2)
        tgt_intr_mtx_inv = np.linalg.inv(tgt_intr_mtx)[:3,:3]
        # fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
        # assert abs(fx - fy) < 1e-9
        # fx = fx * x_scale
        # if focal is None:
        #     focal = fx
        # else:
        #     assert abs(fx - focal) < 1e-5
        pose = tgt_extr_inv_mtx

        pose = (
            self._coord_trans_world
            @ torch.tensor(pose, dtype=torch.float32)
            @ self._coord_trans_cam
        )
        G_tgt_world = pose
        tgt_img_tensor = self.image_to_tensor(tgt_image)
        # print(G_src_world,"SRC")
        # print(G_tgt_world,"TGT")
        G_src_tgt = G_src_world @ np.linalg.inv(G_tgt_world)
        tgt_item = {
            "img":tgt_img_tensor,
            "K":tgt_intr_mtx,
            "K_inv":tgt_intr_mtx_inv,
            "G_src_tgt":G_src_tgt
        }
        # Read tgt items:

        # Sample 3D points in src items
        # TODO: deterministic behavior in val
        # sampled_indices = random.sample(range(len(_src_item["xyzs_ids"])),
        #                                 self.visible_points_count)
        # # sampled_indices = random.sample(range(len(_src_item["xyzs_ids"])),
        # #                                 self.visible_points_count) \
        # #     if not self.is_validation \
        # #     else sorted(range(len(_src_item["xyzs_ids"])))[:256]
        # src_item["xyzs"] = src_item["xyzs"][:, sampled_indices]
        # src_item["xyzs_ids"] = src_item["xyzs_ids"][sampled_indices]
        # src_item["depths"] = src_item["depths"][sampled_indices]
        return src_item, tgt_item

    def __len__(self):
        return len(self.all_imgs)

    def _init_img_transforms(self):
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

   

    def _sample_tgt_items(self, src_idx, src_item,G_src_world,all_cam):
        
        cat,obj_name, _,_= self.all_imgs[src_idx]
        # randomly sample K items for supervision, excluding the src_idx
        scene_indices = [i for i in self.scene_objs_to_indices[cat][obj_name] if i != src_idx]
        # if not self.is_validation:
        sampled_indices = random.sample(scene_indices, self.supervision_count)
        # else:
        #     sampled_indices = [scene_indices[(src_idx + 1) % (len(scene_indices)) - 1]]

        # Generate sampled_items and calculate the relative rotation matrix and translation vector
        # accordingly.
        sampled_items = defaultdict(list)
        for index in sampled_indices:
            cat,obj_name,root_dir,img_dir= self.all_imgs[index]
            img = Image.open(img_dir)
            
            i = int(os.path.basename(img_dir).split(".")[0])
            # img.save(f"tgt_{i}.jpg")
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)
            extr_mtx =  all_cam[wmat_key]
            pose = extr_inv_mtx

            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )
            extr_mtx = pose
            intr_mtx = all_cam["camera_mat_" + str(i)][:3,:3]
            intr_mtx = resize_instrinsic(intr_mtx,2,2)
            intr_mtx_inv = np.linalg.inv(intr_mtx)[:3,:3]


            G_tgt_world = extr_mtx
            G_src_tgt = G_src_world @ np.linalg.inv(G_tgt_world)
            img_tensor = self.image_to_tensor(img)
            sampled_items["img"].append(img_tensor)
            sampled_items["K"].append(intr_mtx)
            sampled_items["K_inv"].append(intr_mtx_inv)
            sampled_items["G_src_tgt"].append(G_src_tgt)

            # Sample xyz points
            # TODO: deterministic behavior in val
            # sampled_xyzs_indices = random.sample(range(len(img_info["xyzs_ids"])),
            #                                      self.visible_points_count) \
            #     if not self.is_validation \
            #     else sorted(range(len(img_info["xyzs_ids"]))[:256])

        return sampled_items


if __name__ == "__main__":
    import logging
    dataset = NeRFDataset({}, logging,
                          root="/scratch/yafathi/NMR_Dataset",
                          is_validation=False,
                          img_size=(128, 128),
                          supervision_count=1,
                          )
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=1, shuffle=False,
                    drop_last=True, num_workers=0)
                    # collate_fn=_collate_fn)
    print(len(dl))
    for batch in dl:
        src_item, supervision_items = batch

        for k, v in src_item.items():
            if(k!="img"):
                print(k, v)

        print("********")

        for k, v in supervision_items.items():
            if(k!="img"):
                print(k, v)

        break
