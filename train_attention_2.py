import argparse
import os
import torch
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
from dataset_combined import RadarDataset
from find_clusters import calc_gmm
from sklearn.cluster import DBSCAN
import time
from random import choices
from radar_scenes.sequence import get_training_sequences, get_validation_sequences, Sequence
from radar_scenes.labels import ClassificationLabel
from radar_scenes.evaluation import per_point_predictions_to_json, PredictionFileSchemas
import models.pointnet2_sem_seg_msg_attention2 as pointnet2_sem_seg_msg_attention
import sys
import time
time_start = time.time()

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'models'))

classes = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP', 'TWO_WHEELER', 'LARGE_VEHICLE ','STATIC']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=10, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=40, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    return parser.parse_args()


def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    checkpoints_dir = "F:/Desktop/bayesianmtl_pointnet"

    NUM_CLASSES = 2
    #NUM_POINT = args.npoint
    dataset = RadarDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # original True
        drop_last=True,
        num_workers=0)
    print("-" * 120)
    print("Mocking a semantic segmentation network...")

    '''MODEL LOADING'''
    MODEL = pointnet2_sem_seg_msg_attention

    #classifier = MODEL.get_model_small(NUM_CLASSES).cpu()
    classifier = MODEL.AttentionSegmodel(NUM_CLASSES).cpu()
    criterion = MODEL.get_loss().cpu()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(checkpoints_dir) + '/model_3.pth')  # model_8 before
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    for epoch in range(1, 2):
        '''Train on chopped scenes'''
        # print('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        # print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)

        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        predTotal = []
        labelTotal = []

        for i, data in enumerate(dataloader, 0):

            print('i:', i)
            optimizer.zero_grad()
            points, target, flow, direction, features = data

            #print(points.shape)
            #print(rcs_dir.shape)
            batch, N = target.shape

            points = np.array(points)
            points = torch.Tensor(points)
            flow = np.array(flow)
            flow = torch.Tensor(flow)

            target = np.array(target)
            target = torch.Tensor(target)
            direction = np.array(direction)
            direction = torch.Tensor(direction)

            features = np.array(features)
            features = torch.Tensor(features)

            points, target = points.float().cpu(), target.long().cpu()
            flow, direction = flow.float().cpu(), direction.long().cpu()
            features = features.float().cpu()

            points = points.transpose(2, 1)
            flow = flow.transpose(2, 1)

            seg_pred, flow_pred = classifier(points, flow)
            # print('seg size1', seg_pred)

            # segmentation loss
            seg_pred = F.log_softmax(seg_pred, dim=1)
            # print('seg size', seg_pred)
            seg_pred = seg_pred.permute(0, 2, 1)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            # features = features.permute(0, 2, 1)
            # features = features.contiguous().view(-1, 6)
            features = features.numpy()

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()  # original
            target = target.view(-1, 1)[:, 0]

            # for binary segmentation
            target[np.where(target != 5)] = 1
            target[np.where(target != 1)] = 0
            loss_seg = criterion(seg_pred, target)

            # direction vector loss
            flow_pred = F.log_softmax(flow_pred, dim=1)
            flow_pred = flow_pred.permute(0, 2, 1)
            flow_pred = flow_pred.contiguous().view(-1, 2)
            direction = direction.view(-1, 1)[:, 0]
            direction[np.where(target != 0)] = 1
            direction[np.where(target != 5)] = 1
            loss_flow = criterion(flow_pred, direction)

            loss = 0.8*loss_seg + 0.5*loss_flow
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            # print('pred_choice', pred_choice, 'size', pred_choice.shape)
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct

            # calculate the number of predicted points in one frame among 8 frames
            frames = pred_choice.reshape(8, 256)

            # load corresponding track_ids & calculate predicted points' track_ids
            check = 'F:/Desktop/bayesianmtl_pointnet/trackids'
            check1 = 'F:/Desktop/bayesianmtl_pointnet/vvids'

            pathsave0 = 'F:/Desktop/seg_labels'
            pathsave1 = 'F:/Desktop/seg_features'
            pathsave2 = 'F:/Desktop/seg_trackids'
            pathsave3 = 'F:/Desktop/seg_vvids'

            for j in range(0, 8):
                segclass = []
                segfeatures = []

                filename = 'track_ids %d' % (i*8+j) + '.npy'
                filepath = os.path.join(check, filename)
                track = np.load(filepath)
                track = track.tolist()

                filename1 = 'vvid %d' % (i*8+j) + '.npy'
                filepath1 = os.path.join(check1, filename1)
                vvid = np.load(filepath1)
                vvid = vvid.tolist()

                frame = frames[j]
                feature = features[j]
                for f in range(len(frame)):
                    if frame[f] == 0:
                        track[f] = 0
                        vvid[f] = 0
                    else:
                        segclass.append(frame[f])
                        segfeatures.append(feature[f, :])

                while 0 in track:
                    track.remove(0)

                while 0 in vvid:
                    vvid.remove(0)

                segclass = np.array(segclass)
                segfeatures = np.array(segfeatures)

                if len(segfeatures) != 0: #and (len(track) != 0):
                    # points in one frame

                    segc = 'label %d' % (i * 8 + j) + '.npy'
                    np.save(os.path.join(pathsave0, segc), segclass)

                    ft = 'feature %d' % (i * 8 + j) + '.npy'
                    np.save(os.path.join(pathsave1, ft), segfeatures)

                    ti = 'track_id %d' % (i*8+j) + '.npy'
                    np.save(os.path.join(pathsave2, ti), track)

                    fnv = 'vvid %d' % (i*8+j) + '.npy'
                    np.save(os.path.join(pathsave3, fnv), vvid)

                    # if len(track_per_frame) != 0 and (len(points_per_frame) > 1):
                    #     if (len(set(track_per_frame)) == 1) and (track_per_frame[0] == b''):
                    #         pass

                else:
                    p = np.array([0,0])
                    t = [0]
                    v = [0]
                    s = [0]

                    segc = 'label %d' % (i * 8 + j) + '.npy'
                    np.save(os.path.join(pathsave0, segc), s)

                    ft = 'feature %d' % (i * 8 + j) + '.npy'
                    np.save(os.path.join(pathsave1, ft), p)

                    ti = 'track_id %d' % (i * 8 + j) + '.npy'
                    np.save(os.path.join(pathsave2, ti), t)

                    fnv = 'vvid %d' % (i * 8 + j) + '.npy'
                    np.save(os.path.join(pathsave3, fnv), v)

            if predTotal == []:
                predTotal = pred_choice
                labelTotal = batch_label
            else:
                predTotal = np.concatenate((predTotal, pred_choice), axis=0)
                labelTotal = np.concatenate((labelTotal, batch_label), axis=0)

            # import sklearn
            # pedID = np.where((labelTotal == 1) | (labelTotal == 2))
            # carID = np.where(labelTotal == 5)
            # BikeID = np.where((labelTotal == 3) | (labelTotal == 4))
            # sklearn.metrics.f1_score(predTotal[pedID], labelTotal[pedID], average='weighted')
            # points[:,2,:].reshape(-1,1)[pedID,0].var()

            total_seen += (args.batch_size * N )
            loss_sum += loss
            print('[epoch: %d] accuracy: %f' % (
                epoch, (correct / (args.batch_size * N ))))

        print('Training mean loss: %f' % (loss_sum / float(total_seen)))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 10 == 0:
            print('Save model...')
            #savepath = str(checkpoints_dir) + '/model_%s.pth'%(epoch)
            savepath = str(checkpoints_dir) + '/model_6.pth'
            print('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            print('Saving model....')
    print("Done with semantic segmentation!")
    print("-" * 120)
    print("\n")
    print("Mocking an instance segmentation network...")

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

if __name__ == '__main__':
    args = parse_args()
    main(args)
