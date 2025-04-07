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
from small_fuc_yuanban import *
from jacobian import *
import glob
# from mask_jacobian import *
from models.segmentation import SegmentationNetwork

def jpg_to_tensor(jpg_path):
    # 打开 jpg 图片
    jpg_image = Image.open(jpg_path)
    jpg_image = jpg_image.resize((256, 256))
    # 定义转换器，将图像转换为 Tensor
    transform = transforms.ToTensor()
    
    # 将图像转换为 Tensor
    jpg_tensor = transform(jpg_image)
    
    return jpg_tensor

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



    def unconditional(self):
        print(self.args.exp)

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
        model = torch.nn.DataParallel(model)
        model.eval()

        # ----------- Precompute Latents -----------#
        seq_inv = np.linspace(0, 1, 999) * 999
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        ###---- boundaries---####
        # ---------- Load boundary ----------#
        classifier = pickle.load(open('./boundary/smile_boundary_h.sav', 'rb'))
        a = classifier.coef_.reshape(1, 512*8*8).astype(np.float32)
        # a = a / np.linalg.norm(a)

        z_classifier = pickle.load(open('./boundary/smile_boundary_z.sav', 'rb'))
        z_a = z_classifier.coef_.reshape(1, 3*256*256).astype(np.float32)
        z_a = z_a / np.linalg.norm(z_a) # normalized boundary                 

        x_lat = torch.randn(1, 3, 256, 256, device=self.device)
        n = 1
        print("get the sampled latent encodings x_T!")

        with torch.no_grad():
            with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
                for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)
                    # print("check t and t_next:", t, t_next)
                    if t == self.args.t_0:
                        break
                    x_lat, h_lat = denoising_step(x_lat, t=t, t_next=t_next, models=model,
                                       logvars=self.logvar,
                                       # sampling_type=self.args.sample_type,
                                       sampling_type='ddim',
                                       b=self.betas,
                                       eta=0.0,
                                       learn_sigma=learn_sigma,
                                       )

                    progress_bar.update(1)




            # ----- Editing space ------ #
            start_distance = self.args.start_distance 
            end_distance = self.args.end_distance
            edit_img_number = self.args.edit_img_number
            linspace = np.linspace(start_distance, end_distance, edit_img_number)
            latent_code = h_lat.cpu().view(1,-1).numpy()
            linspace = linspace - latent_code.dot(a.T)
            linspace = linspace.reshape(-1, 1).astype(np.float32)
            edit_h_seq = latent_code + linspace * a


            z_linspace = np.linspace(start_distance, end_distance, edit_img_number)
            z_latent_code = x_lat.cpu().view(1,-1).numpy()
            z_linspace = z_linspace - z_latent_code.dot(z_a.T)
            z_linspace = z_linspace.reshape(-1, 1).astype(np.float32)
            edit_z_seq = z_latent_code + z_linspace * z_a             


            for k in range(edit_img_number):
                time_in_start = time.time()
                seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                seq_inv = [int(s) for s in list(seq_inv)]
                seq_inv_next = [-1] + list(seq_inv[:-1])

                with tqdm(total=len(seq_inv), desc="Generative process {}".format(it)) as progress_bar:
                    edit_h = torch.from_numpy(edit_h_seq[k]).to(self.device).view(-1, 512, 8, 8)
                    edit_z = torch.from_numpy(edit_z_seq[k]).to(self.device).view(-1, 3, 256, 256)
                    for i, j in zip(reversed(seq_inv), reversed(seq_inv_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        edit_z, edit_h = denoising_step(edit_z, t=t, t_next=t_next, models=model,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta = 1.0,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG,
                                           edit_h=edit_h,
                                           )

                save_edit = "unconditioned_smile_"+str(k)+".png"
                tvu.save_image((edit_z + 1) * 0.5, os.path.join("edit_output",save_edit))
                time_in_end = time.time()
                print(f"Editing for 1 image takes {time_in_end - time_in_start:.4f}s")
        return


    def radius(self):
        print(self.args.exp)

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
        model = torch.nn.DataParallel(model)
        model.eval()


        # ---------- Prepare the seq --------- #

        # seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = np.linspace(0, 1, 999) * 999
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        with torch.no_grad():
            er = 0
            x_rand = torch.randn(100, 3, 256, 256, device=self.device)
            for idx in range(100):
                x = x_rand[idx, :, :, :].unsqueeze(0)

                with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
                    for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        if t == 500:
                            break
                        x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                           logvars=self.logvar,
                                           # sampling_type=self.args.sample_type,
                                           sampling_type='ddim',
                                           b=self.betas,
                                           eta=0.0,
                                           learn_sigma=learn_sigma,
                                           )

                        progress_bar.update(1)
                    r_x = compute_radius(x)

                er += r_x
        print("Check radius at step :", er/100)


        return






    def attri_search(self):
        print(self.args.exp)

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
        model = torch.nn.DataParallel(model)
        model.eval()


        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])


        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, (img, label) in enumerate(loader):
            # for step, img in enumerate(loader):

                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                label = label.to(self.config.device)

                # print("check x and label:", x.size(), label)



                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma,
                                               # edit_h = mid_h,
                                               )

                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone(), mid_h_g.detach().clone(), label])
                    # img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone(), mid_h_g.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Training boundaries -----------#
        print("Start boundary search")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])      


        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            time_in_start = time.time()

            clf_h = svm.SVC(kernel='linear')
            clf_z = svm.SVC(kernel='linear')
            # print("clf model:",clf)

            exp_id = os.path.split(self.args.exp)[-1]
            save_name_h = f'boundary/{exp_id}_{trg_txt.replace(" ", "_")}_h.sav'
            save_name_z = f'boundary/{exp_id}_{trg_txt.replace(" ", "_")}_z.sav'
            n_train = len(img_lat_pairs_dic['train'])
            
            train_data_z = np.empty([n_train, 3*256*256])
            train_data_h = np.empty([n_train, 512*8*8])
            train_label = np.empty([n_train,],  dtype=int)


            for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic['train']):
                train_data_h[step, :] = mid_h.view(1,-1).cpu().numpy()
                train_data_z[step, :] = x_lat.view(1,-1).cpu().numpy()
                train_label[step] = label.cpu().numpy()


            classifier_h = clf_h.fit(train_data_h, train_label)
            classifier_z = clf_z.fit(train_data_z, train_label)
            print(np.shape(train_data_h), np.shape(train_data_z), np.shape(train_label))
            # a = classifier.coef_.reshape(1, 512*8*8).astype(np.float32)
            # a = classifier.coef_.reshape(1, 3*256*256).astype(np.float32)
            # a = a / np.linalg.norm(a)
            time_in_end = time.time()
            print(f"Finding boundary takes {time_in_end - time_in_start:.4f}s")
            print("Finishing boudary seperation!")

            # boudary_save_h = 'smiling_boundary_h.sav'
            # boudary_save_z = 'smiling_boundary_z.sav'
            pickle.dump(classifier_h, open(save_name_h, 'wb'))
            pickle.dump(classifier_z, open(save_name_z, 'wb'))

            # test the accuracy ##
            n_test = len(img_lat_pairs_dic['test'])
            test_data_h = np.empty([n_test, 512*8*8])
            test_data_z = np.empty([n_test, 3*256*256])
            test_lable = np.empty([n_test,], dtype=int)
            for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic['test']):
                test_data_h[step, :] = mid_h.view(1,-1).cpu().numpy()
                test_data_z[step, :] = x_lat.view(1,-1).cpu().numpy()
                test_lable[step] = label.cpu().numpy()
            classifier_h = pickle.load(open(save_name_h, 'rb'))
            classifier_z = pickle.load(open(save_name_z, 'rb'))
            print("Boundary loaded!")
            val_prediction_h = classifier_h.predict(test_data_h)
            val_prediction_z = classifier_z.predict(test_data_z)
            correct_num_h = np.sum(test_lable == val_prediction_h)
            correct_num_z = np.sum(test_lable == val_prediction_z)
            # print(val_prediction_h, test_lable)
            print("Validation accuracy on h and z spaces:", correct_num_h/n_test, correct_num_z/n_test)
            print("total training and testing", n_train, n_test)


        return None




    def edit_image_boundary(self):
        # ----------- Data -----------#
        n = self.args.bs_test


        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img)/255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        # ----------- Models -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
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
        model = torch.nn.DataParallel(model)
        model.eval()

        # ---------- Load boundary ----------#

        boundary_h = pickle.load(open('./boundary/smile_boundary_h.sav', 'rb'))
        a = boundary_h.coef_.reshape(1, 512*8*8).astype(np.float32)
        a = a / np.linalg.norm(a)

        boundary_z = pickle.load(open('./boundary/smile_boundary_z.sav', 'rb'))
        z_a = boundary_z.coef_.reshape(1, 3*256*256).astype(np.float32)
        z_a = z_a / np.linalg.norm(z_a) # normalized boundary


        print("Boundary loaded! In shape:", np.shape(a), np.shape(z_a))


        with torch.no_grad():
            #---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                h_lat_path = os.path.join(self.args.image_folder, f'h_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0,
                                               )


                            progress_bar.update(1)
                        x_lat = x.clone()
                        h_lat = mid_h_g.clone()
                        torch.save(x_lat, x_lat_path)
                        torch.save(h_lat, h_lat_path)

                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)
                    h_lat = torch.load(h_lat_path)
            print("Finish inversion for the given image!", h_lat.size())


            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")


            # ----- Editing space ------ #
            start_distance = self.args.start_distance 
            end_distance = self.args.end_distance
            edit_img_number = self.args.edit_img_number
            # [-100, 100]
            linspace = np.linspace(start_distance, end_distance, edit_img_number)
            latent_code = h_lat.cpu().view(1,-1).numpy()
            linspace = linspace - latent_code.dot(a.T)
            linspace = linspace.reshape(-1, 1).astype(np.float32)
            edit_h_seq = latent_code + linspace * a


            z_linspace = np.linspace(start_distance, end_distance, edit_img_number)
            z_latent_code = x_lat.cpu().view(1,-1).numpy()
            z_linspace = z_linspace - z_latent_code.dot(z_a.T)
            z_linspace = z_linspace.reshape(-1, 1).astype(np.float32)
            edit_z_seq = z_latent_code + z_linspace * z_a           


            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])      

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))


                for k in range(edit_img_number):
                    time_in_start = time.time()

                    with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                        edit_h = torch.from_numpy(edit_h_seq[k]).to(self.device).view(-1, 512, 8, 8)
                        edit_z = torch.from_numpy(edit_z_seq[k]).to(self.device).view(-1, 3, 256, 256)
                        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            edit_z, edit_h = denoising_step(edit_z, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta = 1.0,
                                               learn_sigma=learn_sigma,
                                               ratio=self.args.model_ratio,
                                               hybrid=self.args.hybrid_noise,
                                               hybrid_config=HYBRID_CONFIG,
                                               edit_h=edit_h,
                                               )


                    x0 = x.clone()
                    save_edit = "edited_"+str(k)+".png"
                    tvu.save_image((edit_z + 1) * 0.5, os.path.join("edit_output",save_edit))
                    time_in_end = time.time()
                    print(f"Editing for 1 image takes {time_in_end - time_in_start:.4f}s")
                    

                # this is for recons
                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        x_lat, _ = denoising_step(x_lat, t=t, t_next=t_next, models=model,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           # eta=self.args.eta,
                                           eta = 0.0,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG,
                                           edit_h=None,
                                           )

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                save_edit = "recons.png"
                tvu.save_image((x_lat + 1) * 0.5, os.path.join("edit_output",save_edit))

        return None



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

        if self.config.data.dataset in ["CelebA_HQ"]:
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



        # power_iter = PowerItterationMethod(model)
        # tensor = torch.ones((1, 3, 256, 256))
        # # 将除了 [:, 105:150, 50:205] 部分的所有部分置0
        # # tensor[:, :, :105, :] = 0
        # # tensor[:, :, 150:, :] = 0
        # # tensor[:, :, 105:150, :50] = 0
        # # tensor[:, :, 105:150, 205:] = 0
        # #mouth mask
        # tensor[:, :, :175, :] = 0
        # tensor[:, :, 200:, :] = 0
        # tensor[:, :, 175:200, :95] = 0
        # tensor[:, :, 175:200, 160:] = 0
        # mask =tensor.to("cuda:0")
        
        seg = SegmentationNetwork()
        start_time = time.time()
        directory_path = '/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset'  # 此路径替换为要读取的图像文件夹路径
        jpg_filenames = get_sorted_jpg_filenames(directory_path)
        results_folder = "results/poweriter_mouth_seg/"
        os.makedirs(results_folder, exist_ok=True)
        start_time1 = time.time()
        for filename in jpg_filenames:
            # print(filename)
            orig_image = jpg_to_tensor('/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset/'+filename+'.jpg').unsqueeze(0).to(self.device)
            mask = (seg.get_mask(orig_image, "mouth") ).clamp(0,1).detach()
            mask_img = orig_image*mask
            mask_img = mask_img.squeeze(0)
        # exit()
        #----------------开始计算--------------------#
        #定义保存路径
            # results_folder = "results/poweriter_mouth/"
            # os.makedirs(results_folder, exist_ok=True)
            out_path = results_folder + filename+f"new_num={3}-tol={1e-05}.pt"
            # print(out_path)
            input_latent_path = '/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset_latent/'+ filename + f"_x_lat_t500_ninv40.pth"
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
            input_h_path = '/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset_latent/'+ filename + f"_h_lat_t500_ninv40.pth"
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
                        sval,svec = torch.load('/opt/data/private/lzx/MaskDiffusion/results/poweriter_background_new/'+filename+'new_num=3-tol=1e-05.pt')
                        svals = sval[0][0]
                        edit_h1 = svec[0][0].unsqueeze(0).to(self.device)
                        # print(edit_h1)
                        # print(edit_h1.shape)
                        # exit()
                        if it == 0:
                            edit_h= edit_h +30*edit_h1
                        if it == 1:
                            edit_h= edit_h -30*edit_h1
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

                    save_path = '/opt/data/private/lzx/MaskDiffusion/mouth_results_seg/'
                    #save_path = '/opt/data/private/BoundaryDiffusion-main/'
                    os.makedirs(save_path, exist_ok=True)
                    save_edit = save_path+filename+"jacobian-mouth-"+f"{it}.png"
                    #save_edit = save_path+"255_orig"+".png"   
                    tvu.save_image((edit_z + 1) * 0.5, save_edit)
                    jpg_image = Image.open('/opt/data/private/lzx/MaskDiffusion/selected_dataset/id_dataset/'+filename+'.jpg')
                    png_image = Image.open(save_edit)
                    combine_images(jpg_image, png_image, save_edit)
        print('end!!')
        return None
