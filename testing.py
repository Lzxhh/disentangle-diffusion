import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from sklearn import svm
import pickle
import torch.optim as optim

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment
# from utils.distance_utils import euclidean_distance, cosine_similarity
from small_fuc import *
from jacobian import *
import glob
from models.segmentation import SegmentationNetwork
# from mask_jacobian import *

from PIL import Image
from torchvision.transforms import ToPILImage

def tensor_to_png(tensor, output_path):
    # 确保 Tensor 的范围在 [0, 1]
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # 将 Tensor 转换为 PIL 图像
    to_pil_image = ToPILImage()
    image = to_pil_image(tensor)
    
    # 保存图像为 PNG
    image.save(output_path)

def combine_images(jpg_image, png_image, output_path):
    
    # 确保 png 图片具有 alpha 通道
    if png_image.mode != 'RGBA':
        png_image = png_image.convert('RGBA')
    jpg_image = jpg_image.resize((256, 256))
    # 创建一个新的空白图片，大小为两张图片宽度之和，高度为较大的图片高度
    combined_width = jpg_image.width + png_image.width
    combined_height = max(jpg_image.height, png_image.height)
    combined_image = Image.new('RGBA', (combined_width, combined_height))
    
    # 将 jpg 图片粘贴到新图片的左边
    combined_image.paste(jpg_image, (0, 0))
    
    # 将 png 图片粘贴到新图片的右边
    combined_image.paste(png_image, (jpg_image.width, 0), png_image)
    
    # 将组合图片保存为输出文件
    combined_image.save(output_path)


def jpg_to_tensor(jpg_path):
    # 打开 jpg 图片
    jpg_image = Image.open(jpg_path)
    jpg_image = jpg_image.resize((256, 256))
    # 定义转换器，将图像转换为 Tensor
    transform = transforms.ToTensor()
    
    # 将图像转换为 Tensor
    jpg_tensor = transform(jpg_image)
    
    return jpg_tensor


def get_sorted_jpg_filenames(directory):
    # 获取文件夹中所有jpg文件的路径列表
    jpg_files = glob.glob(os.path.join(directory, '*.jpg'))
    # 按文件名排序
    jpg_files.sort()
    # 提取文件名（不包括路径）
    jpg_filenames = [os.path.splitext(os.path.basename(file))[0] for file in jpg_files]
    return jpg_filenames

def compute_radius(x):
    x = torch.pow(x, 2)
    r = torch.sum(x)
    r = torch.sqrt(r)
    return r




class MaskDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]




    def test(self):
        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load('self.args.model_path')
                # print('load from local path')
                # exit()
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        # model = torch.nn.DataParallel(model)
        model.eval()

        if self.args.n_test_step != 0:
            seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
            seq_test = [int(s) for s in list(seq_test)]
            print('Uniform skip type')
        else:
            seq_test = list(range(self.args.t_0))
            print('No skip')
        seq_test_next = [-1] + list(seq_test[:-1])      


        
        start_time = time.time()
        directory_path = '/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset'  # 此路径替换为要读取的图像文件夹路径
        jpg_filenames = get_sorted_jpg_filenames(directory_path)
        results_folder = "results/poweriter_background_new/"
        os.makedirs(results_folder, exist_ok=True)
        # start_time1 = time.time()
        # out_path = results_folder + '000221'+f"-new_num={3}-tol={1e-05}.pt"
        # out_path = '/opt/data/private/lzx/MaskDiffusion/orig-jacobian-ckpt/poweriter_mouth_orig/google-ddpm-ema-celebahq-256steps10-hspace-afterseed=598212-etas=1num=3-tol=1e-05.pt'
        for filename in jpg_filenames:
        #----------------开始计算--------------------#
            
            input_latent_path = '/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset_latent/'+ filename + f"_x_lat_t500_ninv40.pth"
            input_h_path = '/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset_latent/'+ filename + f"_h_lat_t500_ninv40.pth"
            edit_z = torch.load(input_latent_path)
            edit_h = torch.load(input_h_path)
            # print(1)
            with torch.no_grad():
                # for it in range(0,2):
                edit_z = torch.load(input_latent_path)
                edit_h = torch.load(input_h_path)
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = (torch.ones(1) * i).to(self.device)
                    t_next = (torch.ones(1) * j).to(self.device)
                    # open eyes
                    # sval,svec = torch.load('/opt/data/private/lzx/MaskDiffusion/results/poweriter_eyes/005751-new_num=3-tol=1e-05.pt')
                    sval,svec = torch.load('/opt/data/private/lzx/MaskDiffusion/results/poweriter_mouth/102885-new_num=3-tol=1e-05.pt')
                    #008620 斜嘴
                    sval2,svec2 = torch.load('/opt/data/private/lzx/MaskDiffusion/results/poweriter_mouth/008620-new_num=3-tol=1e-05.pt')
                    svals = sval[0][0]
                    edit_h1 = svec[0][0].unsqueeze(0).to(self.device)
                    # edit_h2 = svec2[0][0].unsqueeze(0).to(self.device)
                    # print(edit_h1)
                    # print(edit_h1.shape)
                    # exit()
                    # if it == 0:
                    # edit_h= edit_h + 50*edit_h1 + 50*edit_h2
                    edit_h= edit_h + 50*edit_h1
                    # if it == 1:
                    #     edit_h= edit_h -30*edit_h1
                    #去噪过程也是DDIM，采用了上面定义的序列(t,tnext)
                    #在中间时间步改变h和z一次，只有一次！！！
                    # print(2)
                    edit_z, edit_h = denoising_step(edit_z, t=t, t_next=t_next, models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta = 0.5,
                                        learn_sigma=learn_sigma,
                                        ratio=self.args.model_ratio,
                                        hybrid=self.args.hybrid_noise,
                                        hybrid_config=HYBRID_CONFIG,
                                        edit_h=edit_h,
                                        )

                # save_path = '/opt/data/private/lzx/MaskDiffusion/orig_results/mouth_edit_results/187-5+_50_0.5/'
                save_path = '/opt/data/private/lzx/MaskDiffusion/transfer_results/mouth_smile_102885/'
                os.makedirs(save_path, exist_ok=True)
                save_edit = save_path+filename+"jacobian-smile"+".png"
                #save_edit = save_path+"255_orig"+".png"   
                tvu.save_image((edit_z + 1) * 0.5, save_edit)
                # jpg_image = Image.open('/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset/'+filename+'.jpg')
                # png_image = Image.open(save_edit)
                # combine_images(jpg_image, png_image, save_edit)
                end_time = time.time()
                print('time:', end_time-start_time)
        print('end!!')
        return None


