import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import scipy.misc

im_src = Image.open("/content/FDA/data_robust/eval/GT/sequence_10_00002.png").convert('RGB')
im_trg = Image.open("/content/FDA/data_robust/eval/noisyblurry/sequence_10_00024.png").convert('RGB')

im_src = im_src.resize( (512,512), Image.BICUBIC )
im_trg = im_trg.resize( (512,512), Image.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))

src_in_trg = np.clip(src_in_trg, 0, 255).astype(np.uint8)
# Konwersja tablicy NumPy do obrazu PIL
image = Image.fromarray(src_in_trg)

# Zapisz obraz
image.save('demo_images/src_in_tar.png')

