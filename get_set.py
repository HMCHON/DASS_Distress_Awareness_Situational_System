'''
import  librarys
'''

from __future__ import print_function, division

import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import sys
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import time
import scipy.io
import yaml
import math
import csv
from torchvision import datasets, models, transforms
from PIL import ImageFont, ImageDraw, Image

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


'''
system path append
'''

sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))
#sys.path.append(os.path.join(os.path.dirname(__file__), ''))

# deep sort
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from detector import build_detector

# re-id
from model import ft_net, ft_net_dense, ft_net_swin, ft_net_NAS, PCB, PCB_test #


'''
parse args setting
'''

def parse_args_deepsort():        
    # Deep sort
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, help="set your video direct") # VIDEO PATH
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--config_mmdetection", type=str, default="configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1.5)
    parser.add_argument("--display_width", type=int, default=1080)
    parser.add_argument("--display_height", type=int, default=720)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--person", action = "store", type = int)

    args = parser.parse_args()

    return args

'''
def parse_args_reid():
    #Fast_reid
    opt = argparse.ArgumentParser(description='Test')
    opt.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    opt.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    opt.add_argument('--test_dir',default='/home/lams/PycharmProjects/torchreid/hm_launcher/RE_ID/Market/pytorch',type=str, help='./test_data')
    opt.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    opt.add_argument('--batchsize', default=256, type=int, help='batchsize')
    opt.add_argument('--use_dense', action='store_true', help='use densenet121' )
    opt.add_argument('--PCB', action='store_true', help='use PCB' )
    opt.add_argument('--multi', action='store_true', help='use multiple query' )
    opt.add_argument('--fp16', action='store_true', help='use fp16.' )
    opt.add_argument('--ibn', action='store_true', help='use ibn.' )
    opt.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = opt.parse_args()
    
    return opt
'''


'''
deep sort part
'''

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
            
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            video_name = self.args.VIDEO_PATH
            video_text_name = str(video_name[:-4] + '.txt')
            video_name = str(video_name[:-4] + '.avi')
            self.save_video_path = os.path.join(self.args.save_path, video_name)
            self.save_results_path = os.path.join(self.args.save_path, video_text_name)

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
    ###########################################################################
    # Set opt dan data_transforms variable
        #opt, ms = opt_setting() # Set Re-identification config
        #data_transforms = Load_Data(opt) # Set Used data config

    ###########################################################################
    # font setting
        fontpath = "./font/hangul_font.ttf"
        font = ImageFont.truetype(fontpath, 30)

    ###########################################################################
    # assert setting
        danger_late = 30 ###  Re-choose this parameter
        check_max_count = 0
        log = []
    ###########################################################################
    # Load Collected data Trained model
        '''
        print('-------test-----------')
        if opt.use_dense:
            model_structure = ft_net_dense(opt.nclasses)
        elif opt.use_NAS:
            model_structure = ft_net_NAS(opt.nclasses)
        elif opt.use_swin:
            model_structure = ft_net_swin(opt.nclasses)
        else:
            model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn )

        if opt.PCB:
            model_structure = PCB(opt.nclasses)

        #if opt.fp16:
        #    model_structure = network_to_half(model_structure)

        model = load_network(opt, model_structure)

        # Remove the final fc layer and classifier layer
        if opt.PCB:
            #if opt.fp16:
            #    model = PCB_test(model[1])
            #else:
                model = PCB_test(model)
        else:
            #if opt.fp16:
                #model[1].model.fc = nn.Sequential()
                #model[1].classifier = nn.Sequential()
            #else:
                model.classifier.classifier = nn.Sequential() 
                
        # Change to test mode
        model = model.eval()
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model = model.cuda()
        '''


    ###########################################################################
    # Deep SORT 

        results = []
        idx_frame = 0
        max_count = 0
        maxmax = 0
        ori_outputs = []
        danger = 0
        NP = self.args.person


        while self.vdo.grab():
            idx_frame += 1
            print(idx_frame)
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking


            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization

            current_count = 0


            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]

                ori_im = draw_boxes(ori_im, bbox_xyxy, identities) # count Add


                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

                # idx_frame = frame number
                # bbox_tlwh = bounding box ((x1,y1,x2,y2),(x1,y1,x2,y2),...,)
                # identities = [1 3 4]


            ###########################################################################
            # when Bounding BOX can't tag target
            '''
            target_th = 30
            for i in range(len(ori_outputs)):
                ori = ori_outputs[i]
                for j in range(len(outputs)):
                    new = outputs[j]
                if ori[4] == new[4]: #ID same
                    if abs(ori[0] - new[0]) > target_th \
                            or abs(ori[1] - new[1]) > target_th \
                            or abs(ori[2] - new[2]) > target_th \
                            or abs(ori[3] - new[3]) > target_th:

                        print(outputs)
                        x1, y1, x2, y2, id = ori[0], ori[1], ori[2], ori[3], ori[4]
                        arr = np.array([[x1, y1, x2, y2, id]])


                        xx = 0.5 * x1 + 0.5 * x2
                        yy = 0.5 * y1 + 0.5 * y2

                        xyoffset = 30

                        xx1 = int(xx - xyoffset)
                        xx2 = int(xx + xyoffset)
                        yy1 = int(yy - xyoffset)
                        yy2 = int(yy + xyoffset)
                        #cv2.rectangle(ori_im, (xx1,yy1), (xx2,yy2), (0, 0, 255), 2)
                        arr = np.array([[x1, y1, x2, y2, id]])
                        outputs[j] = arr
                        print(outputs)
            '''



            ###########################################################################
            # 추적 대상을 놓쳤을 때, 보정 알고리즘 및 조난 상황 알림 발생 알고리즘
            if not len(outputs) == NP:
                danger += 1
                print('danger late = ', danger)

                if len(outputs) == 0:
                    outputs = np.zeros([0, 5])

                check_list = np.zeros(len(ori_outputs)) # 0 = False, 1 = True
                check_list = check_list.tolist()
                threshold = 50

                for i in range(len(ori_outputs)):
                    ori = ori_outputs[i] # [706 154 885 449   5]
                    for j in range(len(outputs)):
                        new = outputs[j] # [706 159 885 454   5]
                        if (ori[0] - threshold) <= new[0] <= (ori[0] + threshold) \
                                and (ori[1] - threshold) <= new[1] <= (ori[1] + threshold) \
                                and (ori[2] - threshold) <= new[2] <= (ori[2] + threshold) \
                                and (ori[3] - threshold) <= new[3] <= (ori[3] + threshold) \
                                or ori[4] == new[4]:  # threshold 범위 안에 new 값이 있다면 또는 같은 ID를 가지고 있다면 동일한 bounding box로 인식
                            check_list[i] = j+1


                for index, value in enumerate(check_list):
                    if value == 0:
                        x1, y1, x2, y2, id = ori_outputs[index][0], ori_outputs[index][1], ori_outputs[index][2], ori_outputs[index][3], ori_outputs[index][4]
                        arr = np.array([[x1, y1, x2, y2, id]])
                        outputs = np.append(outputs, arr, axis = 0)

                        xx = 0.5 * x1 + 0.5 * x2
                        yy = 0.5 * y1 + 0.5 * y2

                        xyoffset = 30

                        xx1 = int(xx - xyoffset)
                        xx2 = int(xx + xyoffset)
                        yy1 = int(yy - xyoffset)
                        yy2 = int(yy + xyoffset)

                        if danger <= danger_late:
                            cv2.rectangle(ori_im, (xx1,yy1), (xx2,yy2), (255, 0, 0), 2)

            else:
                danger = 0

            ###########################################################################
            # 계측 화면 구성 (추적인원수)
            #if danger > danger_late:
                #cv2.putText(ori_im, str(danger), (800, 600), 0, 1, (0, 0, 0), 3)

            if danger < danger_late or idx_frame < 500:
                max_count = len(ori_outputs)
                if max_count > maxmax:
                    maxmax == max_count

            if check_max_count == 0:
                if max_count > NP:
                    max_count = NP
                if max_count < maxmax:
                    max_count = maxmax

            if danger > danger_late and idx_frame > 18000: #make blink alert
                if check_max_count == 1:
                    check_max_count = 0
                else:
                    check_max_count = 1
                    max_count = NP - 1

            cv2.putText(ori_im, str(max_count), (200, 75), 0, 1, (0, 0, 0), 3)
            ori_im = Image.fromarray(ori_im)
            draw = ImageDraw.Draw(ori_im)
            draw.text((10, 50), "추적 인원수 : ", font=font, fill=(0, 0, 0, 0))


            if check_max_count > 0:
                draw.text((850, 50), "조난 상황 발생", font=font, fill=(0, 0, 255, 0))

            ori_im = np.array(ori_im)

            ###########################################################################
            # deep sort end

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            #filename = self.save_video_path
            #write_results(self.save_results_path, results, 'mot')

            # logging
            # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             # .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
            sym = 'X'
            for i in range(len(outputs)):
                if check_max_count == 1:
                    sym = "O"
                self.logger.info("frame: {:.03f}, label: {}, x1: {}, x2: {}, y1: {}, y2: {}, alert: {}" \
                             .format(int(idx_frame), outputs[i][4], outputs[i][0], outputs[i][2], outputs[i][1], outputs[i][3], sym))

                # Save log data
                fields = ['frame', 'label', 'x1', 'x2', 'y1', 'y2', 'alert']
                rows = [idx_frame, outputs[i][4], outputs[i][0], outputs[i][2], outputs[i][1], outputs[i][3], sym]
                log.append(rows)

            ori_outputs = outputs

        name = str(self.args.VIDEO_PATH)
        name = 'output/' + str(name[:-4] + '.csv')

        with open(name, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(log)



'''
re-id part
'''
def opt_setting():
    
    opt = parse_args_reid()   
    
    config_path = os.path.join('./model',opt.name,'opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
    opt.fp16 = config['fp16'] 
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.use_NAS = config['use_NAS']
    opt.use_swin = config['use_swin']
    opt.stride = config['stride']

    if 'nclasses' in config: # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else: 
        opt.nclasses = 751 

    if 'ibn' in config:
        opt.ibn = config['ibn']        
    
    str_ids = opt.gpu_ids.split(',')
    name = opt.name
    test_dir = opt.test_dir
    
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)
    
    print('We use the scale: %s'%opt.ms)
    str_ms = opt.ms.split(',')
    ms = []
    for s in str_ms:
        s_f = float(s)
        ms.append(math.sqrt(s_f))
    
    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    
    return opt, ms


def Load_Data(opt):
    opt = opt
    if opt.use_swin:
        h,w = 224, 224
    else:
        h, w = 256, 128
    
    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    
    return data_transforms
    
def dataloader(opt, directory, data_transforms, frame):
    
    opt = opt
    data_transforms = data_transforms

    #gallery_frame = os.path.join(os.path.join(directory, 'gallery'), frame)
    #query_frame = os.path.join(os.path.join(directory, 'query'), frame-1)

    image_datasets = {x: datasets.ImageFolder( os.path.join(directory,x) ,data_transforms)
                      for x in ['gallery' +'/'+ str(frame), 'query' +'/'+ str(frame-1)]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,shuffle=False, num_workers=16)
                   for x in ['gallery' +'/'+ str(frame), 'query' +'/'+ str(frame-1)]}
        
    return image_datasets, dataloaders
    
###############################################################################
# load model
def load_network(opt, network):
    name = opt.name
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

###############################################################################
# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders,opt,ms):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

###############################################################################
# get id from ./Market/pytorch/~
def get_id(img_path): ######################################cam과 label이 evaluate에 어떻게 연결되는지 찾아서, 가장 ranking이 높은 index를 return하는 함수를 만들어야함.
    camera_id = []
    labels = []

    for path, v in img_path:
        filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:-4] # filename = 123.jpg, label = 123

        camera = '1' # camera angle이 하나이므로 camera = 1로 설정

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels
    
#######################################################################
# Evaluate
def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    # query_index = np.argwhere(gl == ql)
    # camera_index = np.argwhere(gc == qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    #junk_index1 = np.argwhere(gl == -1)
    #junk_index2 = np.intersect1d(query_index, camera_index)
    #junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    '''
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    '''

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


#######################################################################
#######################################################################

if __name__ == "__main__":
    args = parse_args_deepsort()
    cfg = get_config()
    
    if args.mmdet:
        cfg.merge_from_file(args.config_mmdetection)
        cfg.USE_MMDET = True
    else:
        cfg.merge_from_file(args.config_detection)
        cfg.USE_MMDET = False
    
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()

import os
os.system('/home/lams/PycharmProjects/torchreid/hm_launcher/deep_sort_hm/get_set.py deep_sort_hm/test.mp4 --display')





