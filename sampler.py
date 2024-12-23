#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27


import os
import random
import numpy as np
from math import ceil
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte
from ResizeRight.resize_right import resize

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from basicsr.utils import img2tensor
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from facelib.utils.face_restoration_helper import FaceRestoreHelper

class BaseSampler:
    def __init__(self, configs):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/sample/
        '''
        self.configs = configs
        self.display = configs.display
        self.diffusion_cfg = configs.diffusion

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()    # setup seed

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        seed += (self.rank+1) * 10000
        if self.rank == 0 and self.display:
            print(f'Setting random seed {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        gpu_id = self.configs.gpu_id if gpu_id is None else gpu_id
        if gpu_id:
            gpu_id = gpu_id
            num_gpus = len(gpu_id)
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([gpu_id[ii] for ii in range(num_gpus)])
        else:
            num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', init_method='env://')

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def build_model(self):
        obj = util_common.get_obj_from_str(self.configs.diffusion.target)
        self.diffusion = obj(**self.configs.diffusion.params)

        obj = util_common.get_obj_from_str(self.configs.model.target)
        model = obj(**self.configs.model.params).cuda()
        if not self.configs.model.ckpt_path is None:
            self.load_model(model, self.configs.model.ckpt_path)
        self.model = DDP(model, device_ids=[self.rank,]) if self.num_gpus > 1 else model
        self.model.eval()

    def load_model(self, model, ckpt_path=None):
        if not ckpt_path is None:
            if self.rank == 0 and self.display:
                print(f'Loading from {ckpt_path}...')
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            util_net.reload_model(model, ckpt)
            if self.rank == 0 and self.display:
                print('Loaded Done')

    def reset_diffusion(self, diffusion_cfg):
        self.diffusion = create_gaussian_diffusion(**diffusion_cfg)

class DiffusionSampler(BaseSampler):
    def sample_func(self, start_timesteps=None, bs=4, num_images=1000, save_dir=None):
        if self.rank == 0 and self.display:
            print('Begining sampling:')
            save_dir = f'./sample_results' if save_dir is None else save_dir
            util_common.mkdir(save_dir, delete=True)
        if self.num_gpus > 1:
            dist.barrier()

        h = w = self.configs.im_size
        total_iters = ceil(num_images / (bs * self.num_gpus))
        for ii in range(total_iters):
            if self.rank == 0 and self.display:
                print(f'Processing: {ii+1}/{total_iters}')
            noise = torch.randn((bs, 3, h, w), dtype=torch.float32).cuda()
            if 'ddim' in self.configs.diffusion.params.timestep_respacing:
                sample = self.diffusion.ddim_sample_loop(
                        self.model,
                        shape=(bs, 3, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        eta=0.0,
                        )
            else:
                sample = self.diffusion.p_sample_loop(
                        self.model,
                        shape=(bs, 3, h, w),
                        noise=noise,
                        start_timesteps=start_timesteps,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None,
                        device=None,
                        progress=False,
                        )
            sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)
            if save_dir is not None:
                self.imwrite_batch(sample, save_dir, ii+1)

        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            self.tidy_save(save_dir, num_images)

        return sample

    def tidy_save(self, save_dir, num_images):
        files_path = [x for x in Path(save_dir).glob('*.png')]
        if len(files_path) > num_images:
            for path in files_path[num_images:]:
                path.unlink()
        for ii, path in enumerate(files_path[:num_images]):
            new_path = str(path.parent / f'{ii+1}.png')
            os.system(f'mv {path} {new_path}')

    def imwrite_batch(self, sample, fake_dir, bs_ind):
        if not isinstance(fake_dir, Path):
            fake_dir = Path(fake_dir)
        for jj in range(sample.shape[0]):
            im = rearrange(sample[jj,].cpu().numpy(), 'c h w -> h w c') # [0, 1], RGB
            im_path = fake_dir / f'rank{self.rank}_bs{bs_ind}_{jj+1}.png'
            util_image.imwrite(im, im_path, chn='rgb', dtype_in='float32')

class DifIRSampler(BaseSampler):
    def build_model(self):
        super().build_model()

        if not self.configs.model_ir is None:
            obj = util_common.get_obj_from_str(self.configs.model_ir.target)
            model_ir = obj(**self.configs.model_ir.params).cuda()
            if not self.configs.model_ir.ckpt_path is None:
                self.load_model(model_ir, self.configs.model_ir.ckpt_path)
            if self.num_gpus > 1 and len(list(model_ir.parameters(0))) > 0:
                self.model_ir = DDP(model_ir, device_ids=[self.rank,])
            else:
                self.model_ir = model_ir
            self.model_ir.eval()

    def sample_func_ir_aligned(
            self,
            y0,
            start_timesteps=None,
            post_fun=None,
            model_kwargs_ir=None,
            need_restoration=True,
            filter_dict=None,
            end_timesteps=None,
            reg_end_timesteps=None,
            ):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [0, 1], RGB
                or, h x w x c, numpy array, [0, 255], uint8, BGR
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 'ddim250'), range [0, 249]
            post_fun: post-processing for the enhanced image
            model_kwargs_ir: additional parameters for restoration model
        Output:
            sample: n x c x h x w, torch tensor, [0,1], RGB
        '''
        if not isinstance(y0, torch.Tensor):
            y0 = img2tensor(y0, bgr2rgb=True, float32=True).unsqueeze(0) / 255.  # 1 x c x h x w, [0,1]

        if start_timesteps is None:
            start_timesteps = self.diffusion.num_timesteps

        if post_fun is None:
            post_fun = lambda x: util_image.normalize_th(
                    im=x,
                    mean=0.5,
                    std=0.5,
                    reverse=False,
                    )

        # basical image restoration
        device = next(self.model.parameters()).device
        y0 = y0.to(device=device, dtype=torch.float32)
        if need_restoration:
            with torch.no_grad():
                if model_kwargs_ir is None:
                    im_hq = self.model_ir(y0)
                else:
                    im_hq = self.model_ir(y0, **model_kwargs_ir)
        else:
            im_hq = y0
        im_hq.clamp_(0.0, 1.0)
        
        returned_im_hq = im_hq.clone().detach()
        
        h_old, w_old = im_hq.shape[2:4]
        if not (h_old == self.configs.im_size and w_old == self.configs.im_size):
            im_hq = resize(im_hq, out_shape=(self.configs.im_size,) * 2).to(torch.float32)

        # diffuse for im_hq
        yt = self.diffusion.q_sample(
                x_start=post_fun(im_hq),
                t=torch.tensor([start_timesteps,]*im_hq.shape[0], device=device),
                )
        
        if filter_dict is not None:
            x_start = post_fun(im_hq).clone().detach()
            filter_dict['ref_img'] = x_start

        assert yt.shape[-1] == self.configs.im_size and yt.shape[-2] == self.configs.im_size
        if 'ddim' in self.configs.diffusion.params.timestep_respacing:
            sample = self.diffusion.ddim_sample_loop(
                    self.model,
                    shape=yt.shape,
                    noise=yt,
                    start_timesteps=start_timesteps,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                    device=None,
                    progress=False,
                    eta=0.0,
                    )
        else:
            sample = self.diffusion.p_sample_loop(
                    self.model,
                    shape=yt.shape,
                    noise=yt,
                    start_timesteps=start_timesteps,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                    device=None,
                    progress=False,
                    filter_dict=filter_dict,
                    end_timesteps=end_timesteps,
                    reg_end_timesteps=reg_end_timesteps,
                    )

        sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)

        if not (h_old == self.configs.im_size and w_old == self.configs.im_size):
            sample = resize(sample, out_shape=(h_old, w_old)).clamp(0.0, 1.0)

        return sample, returned_im_hq

    def restore_func_ir_aligned(
            self,
            y0,
            model_kwargs_ir=None,
            ):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [0, 1], RGB
                or, h x w x c, numpy array, [0, 255], uint8, BGR
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 'ddim250'), range [0, 249]
            post_fun: post-processing for the enhanced image
            model_kwargs_ir: additional parameters for restoration model
        Output:
            sample: n x c x h x w, torch tensor, [0,1], RGB
        '''
        if not isinstance(y0, torch.Tensor):
            y0 = img2tensor(y0, bgr2rgb=True, float32=True).unsqueeze(0) / 255.  # 1 x c x h x w, [0,1]

        # basical image restoration
        device = next(self.model.parameters()).device
        y0 = y0.to(device=device, dtype=torch.float32)
        if model_kwargs_ir is None:
            im_hq = self.model_ir(y0)
        else:
            im_hq = self.model_ir(y0, **model_kwargs_ir)
        im_hq.clamp_(0.0, 1.0)

        return im_hq


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--gpu_id",
            type=str,
            default='',
            help="GPU Index, e.g., 025",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default='./configs/sample/iddpm_ffhq256.yaml',
            help="Path of config files",
            )
    parser.add_argument(
            "--bs",
            type=int,
            default=32,
            help="Batch size",
            )
    parser.add_argument(
            "--num_images",
            type=int,
            default=3000,
            help="Number of sampled images",
            )
    parser.add_argument(
            "--timestep_respacing",
            type=str,
            default='1000',
            help="Sampling steps for accelerate",
            )
    args = parser.parse_args()

    configs = OmegaConf.load(args.cfg_path)
    configs.gpu_id = args.gpu_id
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    sampler_dist = DiffusionSampler(configs)

    sampler_dist.sample_func(
            bs=args.bs,
            num_images=args.num_images,
            save_dir=args.save_dir,
            )

