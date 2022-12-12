import torchvision.transforms as transforms
import torch
from data.base_dataset import get_params, get_transform, BaseDataset
from PIL import Image
from data.image_folder import make_dataset
from util.preprocessing import folders_image_align, mask_image_generate
import os
import pdb


class TestImageDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--list_dir', type=str, required=False,
                            help='path to the directory that contains photo images')
        parser.add_argument('--raw_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--mask_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--output_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        return parser

    def initialize(self, opt):
        self.opt = opt
        if not os.path.exists(opt.output_dir):
            os.mkdir(opt.output_dir)

        image_paths, mask_paths, output_paths = self.get_paths(opt)

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_paths = output_paths

        size = len(self.image_paths)
        self.dataset_size = size
        transform_list = [
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                ]
        self.image_transform = transforms.Compose(transform_list)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
            ])

    def get_paths(self, opt):
        
        folders_image_align(opt.raw_image_dir, opt.image_dir)
        mask_image_generate(opt.image_dir, opt.mask_dir)
        
        img_names = [file for file in os.listdir(opt.image_dir) 
                     if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")]
        msk_names = [file for file in os.listdir(opt.mask_dir) 
                     if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")]
        img_names.sort()
        msk_names.sort()
        

        image_paths = [f"{opt.image_dir}/{n}" for n in img_names]
        output_paths = [f"{opt.output_dir}/{n}" for n in img_names]
        mask_paths = [f"{opt.mask_dir}/{n}" for n in msk_names]

        return image_paths, mask_paths, output_paths

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        output_path = self.output_paths[index]
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        w, h = image.size
        image_tensor = self.image_transform(image)
        # mask image
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = mask.convert("L")
        mask = mask.resize((w,h))
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor>0).float()
        input_dict = {
                      'image': image_tensor,
                      'mask': mask_tensor,
                      'path': output_path,
                      }

        return input_dict
