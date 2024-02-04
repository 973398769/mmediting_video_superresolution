
import argparse
import os
import glob
import os.path as osp
from tqdm import tqdm
import mmcv
import numpy as np
import cv2
from tqdm import tqdm
import torch
import imageio

from mmedit.datasets.pipelines import Compose
from mmedit.apis import init_model, restoration_video_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=10,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    """Demo for video restoration models.

    Note that we accept video as input/output, when 'input_dir'/'output_dir' is
    set to the path to the video. But using videos introduces video
    compression, which lowers the visual quality. If you want actual quality,
    please save them as separate images (.jpg).
    """

    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)
    device = next(model.parameters()).device
    
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline
    
    tmp_pipeline = []
    for pipeline in test_pipeline:
        if pipeline['type'] not in [
                'GenerateSegmentIndices', 'LoadImageFromFileList'
        ]:
            tmp_pipeline.append(pipeline)

    test_pipeline = tmp_pipeline

    test_pipeline = Compose(test_pipeline)

    input_dir = args.input_dir
    video_reader = mmcv.VideoReader(input_dir)
    frame_count = video_reader.frame_cnt
    fps = video_reader.fps
    width = video_reader.width
    height = video_reader.height
    if width >= 5000 or height >= 2800:
        print("This is already a 5K video.")
        output_dir = f"{input_dir}.out.mp4"
        os.system(f"cp {input_dir} {output_dir}")
    else:
        video_reader = imageio.get_reader(args.input_dir)
        # fourcc = cv2.VideoWriter_fourcc('i', 'Y', 'U', 'V')
        #video_writer = cv2.VideoWriter(args.output_dir, fourcc, video_reader.fps, (video_reader.width * 4, video_reader.height * 4))
        with torch.no_grad():
            for i in tqdm(range(0, frame_count, args.max_seq_len)):
                data = dict(lq=[], lq_path=None, key="")
                frames = []
                for j in range(i, min(i+args.max_seq_len, frame_count-1)):
                    frame = video_reader.get_data(j)
                    if frame is None:
                        print("frame j is none", j)
                    else:
                        frames.append(frame)

                for index, frame in enumerate(frames):
                    if frame is None:
                        print("Error in frame:", index)
                    else:
                        flipped_frame = np.flip(frame, axis=2)
                        data["lq"].append(flipped_frame)

                data = test_pipeline(data)
                data = data['lq'].unsqueeze(0)
                try:
                    result = model(lq=data.to(device), test_mode=True)['output'].cpu()[0]
                    print("result count:", len(result))
                    for k,frame in enumerate(result):
                        output = tensor2img(frame)
                        # print("output:", output)
                        #video_writer.write(output.astype(np.uint8))

                        # write image with 8 zeros padding
                        res = cv2.imwrite(osp.join(args.output_dir, f"{i+k:08d}.jpg"), output)
                        print("write res:", res)
                except:
                    print("Error in frame ", i)
                    continue

        # run ffmpeg to convert images to video
        os.system(f"ffmpeg -y -r {fps} -i {osp.join(args.output_dir, '%08d.jpg')} -c:v libx264 -pix_fmt yuv420p -r {fps} {input_dir}.out.mp4")
if __name__ == '__main__':
    main()