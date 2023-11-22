import torch.nn.functional as F
import numpy as np
from options.train_options_source import TrainOptions
from utils.timer import Timer
import os
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from data import CreateValDataLoader
from model import CreateModel
#import tensorboardX
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
from utils import FDA_source_to_target
import scipy.io as sio

IMG_MEAN = np.array((98.77694003,  74.15956312, 65.00406046), dtype=np.float32)#source values
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
CS_weights = np.array( (1.0, 1.0), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)


def main():
    opt = TrainOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}

    model_name = args.source 
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    sourceloader = CreateSrcDataLoader(args)
    sourceloader_iter = iter(sourceloader)
##
    val_loader=CreateValDataLoader(args)
    val_loader_iter=iter(val_loader)
##

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])



    def validate(model, val_loader, class_weights):
        model.eval()  # Ustaw model w tryb walidacji
        total_loss = 0
        with torch.no_grad():  # Wyłącz obliczanie gradientów
            for val_img, val_lbl, _, _ in val_loader:
                val_img, val_lbl = Variable(val_img).cuda(), Variable(val_lbl.long()).cuda()

                val_seg_score = model(val_img, lbl=val_lbl, weight=class_weights, ita=args.ita)  # Forward pass
                loss_seg_val = model.loss_seg  # Oblicz stratę walidacji

                total_loss += loss_seg_val.detach().cpu().numpy()

        
        return total_loss / len(val_loader)

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.cuda()

    # losses to log
    loss = ['loss_seg_src']
    loss_train = 0.0
    loss_train_list = []
    mean_img = torch.zeros(1, 1)
    class_weights = Variable(CS_weights).cuda()

    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad

        src_img, src_lbl, _, _ = next(sourceloader_iter)  #_  .next()                            # new batch sourc                   

        scr_img_copy = src_img.clone()

        if mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W)

        #-------------------------------------------------------------------#

        # 1. source to target, target to target
        src_in_trg = src_img         # src_lbl

        # 2. subtract mean
        src_img = src_in_trg.clone() - mean_img                                 # src, src_lbl
                                   

        #-------------------------------------------------------------------#

        # evaluate and update params #####
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda() # to gpu
        src_seg_score = model(src_img, lbl=src_lbl, weight=class_weights, ita=args.ita)      # forward pass
        loss_seg_src = model.loss_seg                                                # get loss


        loss_all = loss_seg_src    

        loss_all.backward()
        optimizer.step()

        loss_train += loss_seg_src.detach().cpu().numpy()

        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save( model.state_dict(), os.path.join(args.snapshot_dir, '%s_' % (args.source) + str(i+1) + '.pth') )
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) )

            sio.savemat(args.tempdata, {'src_img':src_img.cpu().numpy()})

            loss_train /= args.print_freq
            loss_train_list.append(loss_train)
            sio.savemat( args.matname, {'loss_train':loss_train_list} )
            loss_train = 0.0
        if (i+1) % args.validation_freq == 0:  # args.validation_freq to liczba iteracji po której chcesz wykonać walidację
            val_loss = validate(model, val_loader, class_weights)
            print('Validation loss at iteration %d: %.4f' % (i + 1, val_loss))

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

if __name__ == '__main__':
    main()

