import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc
import os


src_path = "/content/FDA/data_robust/eval/GT/sequence_10_00002.png"
tar_path="/content/FDA/data_robust/blender_t8/seq_t80001.png"

file_name = os.path.basename(tar_path)

im_src = Image.open(src_path).convert('RGB')
im_trg = Image.open(tar_path).convert('RGB')

im_src = im_src.resize( (416,320), Image.BICUBIC )
im_trg = im_trg.resize( (416,320), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))
L=0.02

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=L )

src_in_trg = src_in_trg.transpose((1,2,0))

src_in_trg = np.clip(src_in_trg, 0, 255).astype(np.uint8)
# Konwersja tablicy NumPy do obrazu PIL
image = Image.fromarray(src_in_trg)

# Zapisz obraz
image.save(f'demo_images/{file_name}_L{L}.png')

