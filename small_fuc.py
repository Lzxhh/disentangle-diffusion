import torch
import numpy as np
import torchvision.io as io
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import pickle
import os 
from PIL import Image
import tqdm
from utils.diffusion_utils import get_beta_schedule, denoising_step_jacobian
from jacobian_method import extend, JacobianMode

def project_and_subtract(a, b):
    # 去掉批量维度
    a_flat = a.squeeze()
    b_flat = b.squeeze()
    
    # 计算 b 在 a 上的投影
    dot_product_ab = torch.dot(a_flat, b_flat)
    dot_product_bb = torch.dot(b_flat, b_flat)
    absin = dot_product_ab/dot_product_bb
    final_result = a_flat.clone()
    for i in range(128):
        batch = b_flat[i * 256: (i + 1) *256]
        # 在这里进行批处理操作
        final_result[i * 256: (i + 1) *256] = a_flat[i * 256: (i + 1) *256]-batch * absin
    
    # 恢复原始维度并从 a 中减去投影
    final_result = final_result.unsqueeze(1)
    # result.unsqueeze(0)
    return final_result



def load_image(image_path):

    # 指定图像文件夹路径
    folder_path = image_path

    # 列出文件夹中所有文件
    file_list = os.listdir(folder_path)

    # 创建一个空列表，用于存储文件路径和文件名
    jpg_files = []

    # 筛选jpg图像文件并存储路径和文件名
    for file_name in file_list:
        # 构建图像文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 检查文件是否为jpg格式
        if file_name.lower().endswith('.jpg'):
            # 存储文件路径和文件名
            file_name_no_extension = os.path.splitext(file_name)[0]
            # 存储文件名（去掉扩展名）和文件路径
            jpg_files.append((file_name_no_extension, file_path))

    # 打印列表内容
    # for file_name, file_path in jpg_files:
    #     print("File Name:", file_name)
    #     print("File Path:", file_path)

    return jpg_files



def calc_power_dirs(self=None,model=None,
                        mask = None,
                        num_eigvecs = 3, 
                        tol = 1e-5, prog_bar = True, 
                        xt=None,
                        t=None,t_next=None,
                        max_iters = 50,
                        input_latent_path=None,
                        learn_sigma=False):
    # input_latent_path = '/opt/data/private/lzx/MaskDiffusion/imgs/img1_x_lat_t500_ninv40.pth'
    xt = torch.load(input_latent_path)
    # print(xt.shape)
    # exit()
    xt.requires_grad = False
    
    # t = 500
    #设置时间步
    seq_test = np.linspace(0, 1, 40) * 500
    seq_test = [int(s) for s in list(seq_test)]
    seq_test_next = [-1] + list(seq_test[:-1])    
    t_to_idx = {int(v):k for k,v in enumerate(seq_test)}
    idx = 0
    svals = torch.zeros((num_eigvecs, 10))
    svecs = torch.zeros((num_eigvecs,) + tuple([10, 512, 8, 8]))
    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
        idx = t_to_idx[i]
        # if idx 
        # if idx %2 == 0:
        t = (torch.ones(1) * i).to(self.device)
        t_next = (torch.ones(1) * j).to(self.device)
    # # xt = xt.clone()
    # xt.requires_grad = False
    # op = tqdm(10)
    # v_init = None 
        # svals = torch.zeros((num_eigvecs, 10))
        # svecs = torch.zeros((num_eigvecs,) + tuple([10, 512, 8, 8]))
        tensor = torch.ones((1, 3, 256, 256)).to("cuda:0")
        with torch.no_grad():
            out1= model.compute(x=xt, timesteps=t)
        h = out1.mid_h
        print(out1.xt.shape)
        # print(h.shape)
        # print(mask.shape)
        # exit()
        # if not mask is None:
        #     print(222)
        #     exit()
        def f_(h):
            out = model.compute(x=xt, timesteps=t, edit_h = -h.detach() + h).xt
            # print(out)
            # if learn_sigma:
            #     out, logvar_learned = torch.split(out, out.shape[1] // 2, dim=1)
            if not mask is None:
                out = out*mask
            return out
        
        def f1_(h):
            out2 = model.compute(x=xt, timesteps=t,edit_h = -h.detach() + h).xt
            # if learn_sigma:
            #     out2, logvar_learned = torch.split(out2, out2.shape[1] // 2, dim=1)
            if not mask is None:
                out3 = out2*(tensor-mask)
            return out3  
            
        
        # h = torch.load('/opt/data/private/lzx/MaskDiffusion/imgs/img1_h_lat_t500_ninv40.pth')
        V,S,U = sequential_power_iteration(f_,f1_,h=h,tol = tol,max_iters=max_iters, k = num_eigvecs)
        print(idx)
        # xt = self.sd.diff.reverse_step(out.out, t, xt, eta = q.etas) 

        svals[:,0] = S      
        svecs[:,0] = U
        idx =idx+1
        if idx > 1:
            break
    return svals, svecs


def sequential_power_iteration( 
                            f,f1, h, v_init = None, 
                            tol = 1e-6, k = 3, 
                            max_iters = 150, etas = None):
    us = []
    ss = []
    vs = []
    us1 = []
    ss1 = []
    vs1 = []
    v_perp = None
    v_perp1 = None
    v_init1 = None
    for i in range(k):

        v,s,u = power_iteration_method(f, h, tol = tol, 
                                                    prog_bar=False, 
                                                    max_iters = max_iters,
                                                    v_perp = v_perp,
                                                    v_init = v_init if v_init is None else v_init[:,i-1] 
                                                    )
        # print(1)
        v1,s1,u1 = power_iteration_method(f1, h, tol = tol, 
                                                    prog_bar=False, 
                                                    max_iters = max_iters,
                                                    v_perp = v_perp1,
                                                    v_init = v_init1 if v_init1 is None else v_init1[:,i-1] 
                                                    )
        # print(2)
        
        u_new = u.reshape(1,512*8*8)
        u1_new = u1.reshape(1,512*8*8)
        u_out = project_and_subtract(u_new,u1_new).to(h.device)
        u = u_out.reshape(1,512,8,8)
        vs.append(v)
        ss.append(s)
        us.append(u)
        v_perp = torch.hstack([v.unsqueeze(1) for v in vs])
        v_init = v_perp 
        vs1.append(v1)
        ss1.append(s1)
        us1.append(u1)
        v_perp1 = torch.hstack([v1.unsqueeze(1) for v1 in vs1])
        # print(v_perp.shape)
        # exit()
        v_init1 = v_perp1 
    if len(us[0].shape) == 1:
        us = torch.cat([u.unsqueeze(0) for u in us])
    else: 
        us = torch.cat(us)
    if len(vs[0].shape) == 1:
        vs = torch.cat([v.unsqueeze(0) for v in vs])
    else: 
        vs = torch.cat(vs)
    ss = torch.cat([s.unsqueeze(0) for s in ss])  
    return vs,ss,us




def power_iteration_method(
                                f, h,tol = 1e-7, 
                                max_iters = 50,
                                prog_bar = False,
                                v_perp = None,
                                v_init = None):
        """
                ## steps for power itteration method 
                # (1) sps matrix A  = J.T @ J  
                # (2) v_hat = Av
                # (3) v = v_hat / norm(vhat)
                # (4) Av = J.T @ J v 
                # (5) d/dh < y, h> = Jv = q
                # define h = z + aq (where a is some scaler and z is the diff btw h and a*q)
                # now d/da f(z + aq) = d/da f(h)  = J.T @ q
        """
        # if not v_perp is None: 
        #     v_perp = v_perp.to(h.device)
        #     part1 = v_perp @ torch.linalg.inv(v_perp.T @ v_perp)
        #     part2 = v_perp.T
            # project_and_subtract()

        h.requires_grad = True 

        ## Step 1 (inialize v)
        y = f(h).view(-1)  # flatten 
        # print(y.shape)
        # exit()
        if v_init is None:
            v = torch.randn_like(y) 
            v = v / torch.sqrt(v@v)
        else: 
            v =  v_init.to(h.device)
        tol_counter = 0
        eigval_prev = 999
        op = tqdm(range(max_iters)) if prog_bar else range(max_iters)
        # Jv = torch.autograd.grad( y @ v , h, retain_graph=True)[0].detach().clone()
        # print(Jv)
        # exit()
        
        for iteration in op:   
            ## d/dh < y, h> = Jv = q 
            Jv = torch.autograd.grad( y @ v, h, retain_graph=True)[0].detach().clone()
            #jv.shape = 1, 512, 8, 8
            ## Now calcualte J.T @ q (where q is an arbitrary vector)
            a = torch.ones(1, requires_grad=True, device=h.device)
            z = h - a * Jv
            # a.shape=1
            #z.shape = 1, 512, 8, 8
            z = z.detach().clone() 
            Jv = Jv.contiguous()

            #Calculate  J.T @ J v 
            # f_tilde = lambda a_:  f(z + a_* Jv )
            def f_tilde(a_):
                return f(z + a_ * Jv)
            y2 = f_tilde(a)
            JtJv = torch.autograd.functional.jacobian(f_tilde, a, vectorize=True,  strategy='forward-mode').detach() #create_graph=False, strict=False, vectorize=True, strategy='forward-mode'
            # JtJv = torch.autograd.functional.jacobian(f_tilde, a)
            # with JacobianMode(f_tilde):
            #     out = f_tilde(a)
            #     out.sum().backward()
            #     JtJv = f_tilde.jacobian()

            # JtJv = torch.autograd.grad(y2, a, retain_graph=True)[0].detach().clone()
            #jtjv.shape = 1,3,256,256,1
            v_hat = JtJv.flatten()
            # Projection on the orthogonal complement
            # if not v_perp is None: 
            #     v_hat_projected = part1 @ (part2 @ v_hat.detach())
            #     v_hat = v_hat - v_hat_projected
            
            eigval = (v_hat*v_hat).sum() ** 0.5
            sval = eigval ** 0.5
            v = v_hat / eigval 
            
            err = abs(eigval-eigval_prev)
            eigval_prev = eigval
          
            if err < tol:
                tol_counter += 1
                if tol_counter > 10:
                    break
    
        if iteration + 1 == max_iters:
            print("[WARNING], max iters reached")
            print("final err", err)

        return v, sval, Jv/sval