"""
Author: suhaib mahmood
email: suhaib.mahmud22@gmail.com

"""
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import cv2  # OpenCV import
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
import imageio.v2 as imageio  # Use ImageIO v2 to avoid deprecation warnings

from modules.generator import Generator
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork
from animate import get_animation_region_params
import matplotlib

matplotlib.use('Agg')

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = Generator(
        num_regions=config['model_params']['num_regions'],
        num_channels=config['model_params']['num_channels'],
        **config['model_params']['generator_params']
    )
    generator.to(device)

    region_predictor = RegionPredictor(
        num_regions=config['model_params']['num_regions'],
        num_channels=config['model_params']['num_channels'],
        estimate_affine=config['model_params']['estimate_affine'],
        **config['model_params']['region_predictor_params']
    )
    region_predictor.to(device)

    avd_network = AVDNetwork(
        num_regions=config['model_params']['num_regions'],
        **config['model_params']['avd_network_params']
    )
    avd_network.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    if device.type == 'cuda':
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)
        avd_network = DataParallelWithCallback(avd_network)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()

    return generator, region_predictor, avd_network


def make_animation(source_image, driving_video, generator, region_predictor, avd_network,
                   animation_mode='standard', device=torch.device("cpu")):
    with torch.no_grad():
        predictions = []
        # Prepare source tensor and move it to device
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        # Compute region parameters for the source once
        source_region_params = region_predictor(source)
        
        # Get initial region parameters from the first driving frame
        first_frame = driving_video[0]
        first_frame_resized = torch.tensor(first_frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        driving_region_params_initial = region_predictor(first_frame_resized)
        
        for frame in tqdm(driving_video):
            frame_tensor = torch.tensor(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
            driving_region_params = region_predictor(frame_tensor)
            new_region_params = get_animation_region_params(
                source_region_params,
                driving_region_params,
                driving_region_params_initial,
                avd_network=avd_network,
                mode=animation_mode
            )
            out = generator(source, source_region_params=source_region_params, driving_region_params=new_region_params)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions



def main(opt):
    # Determine device: use GPU if available and not forced to CPU
    device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    
    # Read source image with imageio.v2
    source_image = imageio.imread(opt.source_image)
    
    # Read driving video with OpenCV
    cap = cv2.VideoCapture(opt.driving_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Fallback FPS if not available
    driving_video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        driving_video.append(frame)
    cap.release()

    # Resize images to the shape the model was trained on
    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    
    # Load checkpoints and generate animation using the proper device
    generator, region_predictor, avd_network = load_checkpoints(
        config_path=opt.config,
        checkpoint_path=opt.model,
        device=device
    )
    
    predictions = make_animation(
        source_image,
        driving_video,
        generator,
        region_predictor,
        avd_network,
        animation_mode=opt.mode,
        device=device
    )
    
    # Save the output video using imageio.v2
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='ted384.yaml', help="path to config")
    parser.add_argument("--model", default='ted384.pth', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='data/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='data/drivingv.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='output/result.mp4', help="path to output")
    parser.add_argument("--mode", default='avd', choices=['standard', 'relative', 'avd'], help="Animation mode")
    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help="Shape of image, that the model was trained on.")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="force CPU mode")
    
    main(parser.parse_args())

