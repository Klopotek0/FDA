import numpy as np
from torch.utils import data
from data.gta5_dataset_source import GTA5DataSet1
from data.gta5_dataset import GTA5DataSet

from data.cityscapes_dataset import cityscapesDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.cityscapes_dataset_SSL import cityscapesDataSetSSL
from data.synthia_dataset import SYNDataSet

IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
image_sizes = {'target': (416,320), 'soruce_images': (416, 320), 'synthia': (1280, 760)}
cs_size_test = {'target': (416,320)}

def CreateSrcDataLoader(args):
    if args.source == 'soruce_images':
        source_dataset = GTA5DataSet1( args.data_dir, args.train_list, crop_size=image_sizes['target'], 
                                      resize=image_sizes['soruce_images'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    elif args.source == 'synthia':
        source_dataset = SYNDataSet( args.data_dir, args.data_list, crop_size=image_sizes['target'],
                                      resize=image_sizes['synthia'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    source_dataloader = data.DataLoader( source_dataset, 
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         num_workers=args.num_workers, 
                                         pin_memory=True )    
    return source_dataloader

def CreateValDataLoader(args):
    if args.source == 'soruce_images':
        val_dataset = GTA5DataSet( args.data_dir, args.val_list, crop_size=image_sizes['target'], ##
                                      resize=image_sizes['soruce_images'] ,mean=IMG_MEAN,
                                      max_iters=90 // 1)
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    val_dataloader = data.DataLoader( val_dataset, 
                                         batch_size=1,
                                         shuffle=False, 
                                         num_workers=args.num_workers, 
                                         pin_memory=True )    
    return val_dataloader

def CreateTrgDataLoader(args):
    if args.set == 'train' or args.set == 'trainval':
        target_dataset = cityscapesDataSetLabel( args.data_dir_target, 
                                                 args.data_list_target, 
                                                 crop_size=image_sizes['target'], 
                                                 mean=IMG_MEAN, 
                                                 max_iters=args.num_steps * args.batch_size, 
                                                 set=args.set )
    else:
        target_dataset = cityscapesDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['target'],
                                            mean=IMG_MEAN,
                                            set=args.set )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader



def CreateTrgDataSSLLoader(args):
    target_dataset = cityscapesDataSet( args.data_dir_target, 
                                        args.data_list_target,
                                        crop_size=image_sizes['target'],
                                        mean=IMG_MEAN, 
                                        set=args.set )
    target_dataloader = data.DataLoader( target_dataset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         pin_memory=True )
    return target_dataloader



def CreatePseudoTrgLoader(args):
    target_dataset = cityscapesDataSetSSL( args.data_dir_target,
                                           args.data_list_target,
                                           crop_size=image_sizes['target'],
                                           mean=IMG_MEAN,
                                           max_iters=args.num_steps * args.batch_size,
                                           set=args.set,
                                           label_folder=args.label_folder )

    target_dataloader = data.DataLoader( target_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True )

    return target_dataloader

