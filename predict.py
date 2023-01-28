
from cog import BasePredictor, Input, Path
import os

class Predictor(BasePredictor):
    def setup(self):
        print("setup")
        # os.system("cd /cvpr/codes/models/modules/DCNv2 && bash make.sh")
        os.system("pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html")
        os.system("pip install --upgrade pip")
        os.system("pip install -e .")
        # os.system("mim install mmcv-full")

    def predict(self, 
        video: Path = Input(description="input image"),
        task_type: str = Input(
            description='image restoration task type',
            default='Real-World Image Super-Resolution',
            choices=['Real-World Image Super-Resolution', 'Grayscale Image Denoising', 'Color Image Denoising','JPEG Compression Artifact Reduction']        
        ),  
        scale_factor: int = Input(
            description="updscale factor for RealSR. 2 or 4 are allowed",
            default=4),
        jpeg: int = Input(
            description="scale factor, activated for JPEG Compression Artifact Reduction. ",
            default=40),
        noise: int = Input(
             description="noise level, activated for Grayscale Image Denoising and Color Image Denoising.",
             default=15)
        ) -> Path:
        print("predict")

        # extract frames from video
        os.system(f"mkdir -p output frames")
        os.system("rm -rf output/* frames/*")

        os.system(f"python run_on_video_with_window.py \
                ./configs/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py \
                https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth \
                {str(video)} \
                ./output \
                --max-seq-len=20")

                # --window-size=2 \

        # os.system("ls -l ./output")
        # os.system("ls -l .")



        return Path(f"{str(video)}.out.mp4")

#  python demo/restoration_video_demo.py configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth                terrenas_cut.mov                 ./output                 --window-size=5


## ffmpeg command to resize video to 640x360 with x264 codec

# ffmpeg -i terrenas_cut.mov -vf scale=640:360 -c:v libx264 -crf 23 -preset veryfast terrenas_cut_640x360.mp4