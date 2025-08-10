import argparse
from pathlib import Path
from omegaconf import OmegaConf

from utils import util_image

from sampler import DifIRSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--gpu_id",
            type=str,
            default='0',
            help="GPU Index",
            )
    parser.add_argument(
            "-s",
            "--started_timesteps",
            type=int,
            default='100',
            help='Started timestep for DifFace, parameter N in our paper (Default:100)',
            )
    parser.add_argument(
            "-t",
            "--timestep_respacing",
            type=str,
            default='250',
            help='Sampling steps for Improved DDPM, parameter T in out paper (default 250)',
            )
    parser.add_argument(
            "--in_path",
            type=str,
            default='./testdata/cropped_faces',
            help='Folder to save the low quality image',
            )
    parser.add_argument(
            "--out_path",
            type=str,
            default='./results',
            help='Folder to save the restored results',
            )
    parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help='path to the ckpt to be tested',
            )
    args = parser.parse_args()

    cfg_path = 'configs/targets/iddpm_ffhq512_swinir_gan.yaml'

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = args.gpu_id
    configs.aligned = True
    assert args.started_timesteps < int(args.timestep_respacing)
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # build the sampler for diffusion
    configs.model_ir.ckpt_path = args.ckpt_path
    sampler_dist = DifIRSampler(configs)

    # prepare low quality images
    exts_all = ('jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'bmp')
    if args.in_path.endswith(exts_all):
        im_path_list = [Path(args.in_path), ]
    else: # for folder
        im_path_list = []
        for ext in exts_all:
            im_path_list.extend([x for x in Path(args.in_path).glob(f'*.{ext}')])

    im_path_list = sorted(im_path_list)

    # prepare result path
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)
    restored_face_dir = Path(args.out_path) / 'restored_faces'
    if not restored_face_dir.exists():
        restored_face_dir.mkdir()

    for ii, im_path in enumerate(im_path_list):
        if (ii+1) % 5 == 0:
            print(f"Processing: {ii+1}/{len(im_path_list)}...")
        im_lq = util_image.imread(im_path, chn='bgr', dtype='uint8')

        face_restored = sampler_dist.restore_func_ir_aligned(
                y0=im_lq,
                ) #[0,1], 'rgb'

        face_restored = util_image.tensor2img(
                face_restored,
                rgb2bgr=True,
                min_max=(0.0, 1.0),
                ) # uint8, BGR
        save_path = restored_face_dir / im_path.name
        util_image.imwrite(face_restored, save_path, chn='bgr', dtype_in='uint8')


if __name__ == '__main__':
    main()