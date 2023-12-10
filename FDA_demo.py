import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc
import os


src_path = "" #path to source image
tar_path="" #path to target image

tar_file_name = os.path.basename(tar_path)
src_file_name = os.path.basename(src_path)



im_src = Image.open(src_path).convert('RGB')
im_trg = Image.open(tar_path).convert('RGB')
im_src.save(f'demo_images/{src_file_name}')
im_trg.save(f'demo_images/{tar_file_name}')

im_src = im_src.resize( (416,320), Image.BICUBIC )
im_trg = im_trg.resize( (416,320), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))
L=0.01

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=L )

src_in_trg = src_in_trg.transpose((1,2,0))

src_in_trg = np.clip(src_in_trg, 0, 255).astype(np.uint8)

image = Image.fromarray(src_in_trg)

image.save(f'demo_images/{src_file_name} to {tar_file_name}_L{L}.png')

