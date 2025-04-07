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


from PIL import Image

def combine_images(jpg_image, png_image, output_path):

    if png_image.mode != 'RGBA':
        png_image = png_image.convert('RGBA')
    jpg_image = jpg_image.resize((256, 256))
    combined_width = jpg_image.width + png_image.width
    combined_height = max(jpg_image.height, png_image.height)
    combined_image = Image.new('RGBA', (combined_width, combined_height))
  
    combined_image.paste(jpg_image, (0, 0))

    combined_image.paste(png_image, (jpg_image.width, 0), png_image)
 
    combined_image.save(output_path)


def jpg_to_tensor(jpg_path):
    jpg_image = Image.open(jpg_path)
    jpg_image = jpg_image.resize((256, 256))
    transform = transforms.ToTensor()
    jpg_tensor = transform(jpg_image)
    
    return jpg_tensor


def get_sorted_jpg_filenames(directory):
    jpg_files = glob.glob(os.path.join(directory, '*.jpg'))
    jpg_files.sort()
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




    def compute_latent(self):
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
                init_ckpt = torch.load(self.args.model_path)
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


        # ----------- Precompute Latents -----------#
        # print("Prepare identity latent")
        # seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        # seq_inv = [int(s) for s in list(seq_inv)]
        # seq_inv_next = [-1] + list(seq_inv[:-1])


        # n = self.args.bs_train
        # x_orig_path = '/opt/data/private/BoundaryDiffusion-main/img_label/glasses/000152.jpg'
        # x_lat_path = os.path.join('/opt/data/private/BoundaryDiffusion-main/precomputed', f'152-x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
        # h_lat_path = os.path.join('/opt/data/private/BoundaryDiffusion-main/precomputed', f'152-h_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
        # folder_path = '/opt/data/private/lzx/MaskDiffusion/imgs'
        # print(1)
        # file_list = load_image(folder_path)
        # print(file_list)
        # for file_name, file_path in file_list:
        #     x_lat_path = os.path.join(folder_path, f'{file_name}_x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
        #     h_lat_path = os.path.join(folder_path, f'{file_name}_h_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
        #     seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        #     seq_inv = [int(s) for s in list(seq_inv)]
        #     seq_inv_next = [-1] + list(seq_inv[:-1])
        #     img = Image.open(file_path).convert("RGB")
        #     img = img.resize((256,256), Image.LANCZOS)
        #     img = np.array(img)/255
        #     img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        #     img = img.to(self.config.device)
        #     x0 = (img - 0.5) * 2
        #     x=x0.clone()
        #     with torch.no_grad():
        #         with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
        #             for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
        #                 t = (torch.ones(n) * i).to(self.device)
        #                 t_prev = (torch.ones(n) * j).to(self.device)
        #                 print(t)
        #                 print(t_prev)

        #                 x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model,
        #                                     logvars=self.logvar,
        #                                     sampling_type='ddim',
        #                                     b=self.betas,
        #                                     eta=0,
        #                                     learn_sigma=learn_sigma,
        #                                     ratio=0,
        #                                     )


        #                 progress_bar.update(1)
        #     x_lat = x.clone()
        #     h_lat = mid_h_g.clone()
        #     torch.save(x_lat, x_lat_path)
        #     torch.save(h_lat, h_lat_path)
        #     print(h_lat_path)

        # ----------- Generative Process -----------#
        # print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
        #         f" Steps: {self.args.n_test_step}/{self.args.t_0}")


        if self.args.n_test_step != 0:
            seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
            seq_test = [int(s) for s in list(seq_test)]
            print('Uniform skip type')
        else:
            seq_test = list(range(self.args.t_0))
            print('No skip')
        seq_test_next = [-1] + list(seq_test[:-1])      


        seg = SegmentationNetwork()
        # jpg_image = Image.open('/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset/'+filename+'.jpg')
        # mask = (seg.get_mask(q, "l_eye") ).clamp(0,1).detach()
        # power_iter = PowerItterationMethod(model)
        # tensor = torch.ones((1, 3, 256, 256))
        # tensor[:, :, :105, :] = 0
        # tensor[:, :, 150:, :] = 0
        # tensor[:, :, 105:150, :50] = 0
        # tensor[:, :, 105:150, 205:] = 0
        #mouth mask
        # tensor[:, :, :175, :] = 0
        # tensor[:, :, 200:, :] = 0
        # tensor[:, :, 175:200, :95] = 0
        # tensor[:, :, 175:200, 160:] = 0
        # mask =tensor.to("cuda:0")
        
        
        start_time = time.time()
        directory_path = 'dataset/id_dataset'  # 此路径替换为要读取的图像文件夹路径
        jpg_filenames = get_sorted_jpg_filenames(directory_path)
        results_folder = "results/poweriter_hair/"
        os.makedirs(results_folder, exist_ok=True)
        start_time1 = time.time()
        for filename in jpg_filenames:
            orig_image = jpg_to_tensor('dataset/id_dataset/'+filename+'.jpg').unsqueeze(0).to(self.device)
            mask = (seg.get_mask(orig_image, "hair") ).clamp(0,1).detach()

        #----------------开始计算--------------------#
            out_path = results_folder + filename+f"-new_num={3}-tol={1e-05}.pt"
            # print(out_path)
            input_latent_path = 'dataset/id_dataset_latent/'+ filename + f"_x_lat_t500_ninv40.pth"
            # print(input_latent_path)
            if os.path.exists(out_path):
                svals, svecs = torch.load(out_path)
                # print("[INFO] Loaded from", out_path)
            else: 
                svals, svecs = calc_power_dirs(self,model,prog_bar = True, 
                                               max_iters = 50, mask=mask, tol = 1e-5, num_eigvecs = 3,
                                               input_latent_path = input_latent_path)
                torch.save([svals, svecs], out_path)
                # print("[INFO], saved to", out_path)
            end_time1 = time.time()
            print(f"Time taken: {end_time1 - start_time1:.4f}s")

            # svals, svecs = load_power_dirs(model, mask = mask)
            input_h_path = 'dataset/id_dataset_latent/'+ filename + f"_h_lat_t500_ninv40.pth"
            edit_z = torch.load(input_latent_path)
            edit_h = torch.load(input_h_path)
            # print(1)
            with torch.no_grad():
                for it in range(0,2):
                    edit_z = torch.load(input_latent_path)
                    edit_h = torch.load(input_h_path)
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(1) * i).to(self.device)
                        t_next = (torch.ones(1) * j).to(self.device)
                        sval,svec = torch.load('/results/poweriter/'+filename+'-new_num=3-tol=1e-05.pt')
                        svals = sval[0][0]
                        edit_h1 = svec[0][0].unsqueeze(0).to(self.device)

                        if it == 0:
                            edit_h= edit_h +30*edit_h1
                        if it == 1:
                            edit_h= edit_h -30*edit_h1

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

                    save_path = 'your_save_path/'  # 替换为你想要保存的路径
                    os.makedirs(save_path, exist_ok=True)
                    save_edit = save_path+filename+"jacobian"+f"{it}.png"  
                    tvu.save_image((edit_z + 1) * 0.5, save_edit)
                    jpg_image = Image.open('your_orig_img_path/'+filename+'.jpg')
                    png_image = Image.open(save_edit)
                    combine_images(jpg_image, png_image, save_edit)
        print('end!!')
        return None
