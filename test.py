from model import DG_rPPGNet
from utils_sig import *
import argparse

from dataloader import get_loader
from loss import *

from model import rPPG_estimator, Encoder, Separator, Classifier, Decoder_video, GradReverse, Difficult_Transform, Project_Head
from load_save_model import load_model, save_model
from util import *
from dataloader import get_loader


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

else:
    print("no gpu")
    device = torch.device('cpu')


args = get_args()
trainName, testName = get_name(args)
log = get_logger(f"logger/test/{args.test_dataset}/", testName)

print(f"{trainName=}")
print(f"{testName=}")

class_num = 2
domain_order = [1, 0]
medium_ch = 16


seq_len = args.test_T*args.fps
print(seq_len)
stride = seq_len
test_loader = get_loader(_datasets=list(args.test_dataset), 
                        _seq_length=seq_len,
                        batch_size=1,
                        shuffle=False,
                        train=False,
                        if_bg=False,)


conv_type='vanilla'

feature_extractor = Encoder(medium_channels=medium_ch, task="global").to(device)
separator_rPPG = Separator(in_ch=medium_ch, out_ch=medium_ch, task="rPPG").to(device)
estimator_rPPG = rPPG_estimator(in_ch=medium_ch, seq_length=seq_len).to(device)
estimator_rPPG_G = rPPG_estimator(in_ch=medium_ch, seq_length=seq_len).to(device)


with torch.no_grad():

    for epoch in range (..., args.epoch):

        load_idx = epoch - 400
        #load_idx = -1 : mean the newest(last) model in your folder
        feature_extractor, epoch_F = load_model(
            "feature_extractor", device, feature_extractor, train_mode=False, load_idx=load_idx)
        feature_extractor.eval()
        separator_rPPG, epoch_S_rppg = load_model(
            "separator_rPPG", device, separator_rPPG, train_mode=False, load_idx=load_idx)
        separator_rPPG.eval()
        estimator_rPPG, epoch_E_rppg = load_model(
            "estimator_rPPG", device, estimator_rPPG, train_mode=False, load_idx=load_idx)
        estimator_rPPG.eval()
        estimator_rPPG_G, epoch_E_rppg_G = load_model(
            "estimator_rPPG_G", device, estimator_rPPG_G, train_mode=False, load_idx=load_idx)
        estimator_rPPG_G.eval()


        print("epoch : ", epoch_F)

        all_mae = []
        all_rmse = []
        all_R = []
        all_mae2 = []
        all_rmse2 = []
        all_R2 = []
        for step, (face_frames, image_paths, ppg_label, bg_frames, bg_paths, domain_labels, id_labels) in enumerate(test_loader):
            
            print(face_frames.shape)
            
            imgs = face_frames.squeeze(1)#.to(device)
            # imgs = face_frames.squeeze(1)#.to(device)
            print(image_paths[0][0], imgs.shape, ppg_label.shape)
            label_PPG = ppg_label.squeeze(1)#.to(device)
            hr_predicts = []
            hr_predicts2 = []
            hr_labels = []
            hr_labels2 = []

            for i in range(0, imgs.shape[2] - seq_len + stride, stride):
                _imgs = imgs[:, :, i:i + seq_len, :, :].to(device)
                _label = label_PPG[:, i:i + seq_len].detach().cpu().numpy()
    
                #global_rPPG, disent_rPPG = model(face_frames, label_rPPG, label_id, label_domain, domain_order, learning_type=-1)
                
                global_feat = feature_extractor(_imgs)
                global_rPPG = estimator_rPPG_G(global_feat)
                rppg = global_rPPG[0].detach().cpu().numpy()

                rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=args.fps)
                
                # get_Plot(rppg.copy(), name=f"bg_only_{step}_{i}_overall", subj=image_paths[0][0])
                # exit()
                
                _label = butter_bandpass(_label, lowcut=0.6, highcut=4, fs=args.fps)

                hr_rppg, psd_y_rppg, psd_x_rppg = hr_fft(rppg.copy(), fs=args.fps)
                hr_label, psd_y_label, psd_x_label = hr_fft(_label, fs=args.fps)
                hr_predicts.append(hr_rppg)
                hr_labels.append(hr_label)
                
                hr_rppg2 = predict_heart_rate(rppg.copy(), args.fps)
                hr_label2 = predict_heart_rate(_label, args.fps)
                hr_predicts2.append(hr_rppg2)
                hr_labels2.append(hr_label2)

            # exit(0)
            hr_predicts = np.array(hr_predicts)
            hr_labels = np.array(hr_labels)
            hr_predicts2 = np.array(hr_predicts2)
            hr_labels2 = np.array(hr_labels2)

            print(f"{hr_predicts=}")
            print(f"{hr_labels=}")
            print(f"{hr_predicts2=}")
            print(f"{hr_labels2=}")

            mae = np.mean(np.abs(hr_predicts - hr_labels))
            rmse = np.sqrt(np.mean((hr_predicts - hr_labels) ** 2))

            pearson_corr = Pearson_np(hr_predicts, hr_labels)
    
            log.info('(1) [epoch %d step %d mae %.5f rmse %.5f pearson_corr %.5f]' 
                        % (epoch, step, mae, rmse, pearson_corr))
            
            all_mae.append(mae)
            all_rmse.append(rmse)
            all_R.append(pearson_corr)
            
            mae2 = np.mean(np.abs(hr_predicts2 - hr_labels2))
            rmse2 = np.sqrt(np.mean((hr_predicts2 - hr_labels2) ** 2))

            pearson_corr2 = np.corrcoef(hr_predicts2, hr_labels2)[0,1]
            
            log.info('(2) [epoch %d step %d mae %.5f rmse %.5f pearson_corr %.5f]' 
                        % (epoch, step, mae2, rmse2, pearson_corr2))
            
            all_mae2.append(mae2)
            all_rmse2.append(rmse2)
            all_R2.append(pearson_corr2)
            
        log.info('(1) [epoch %d avg all_mae %.5f all_rmse %.5f all_R %.5f]'
                    % (epoch, np.mean(all_mae), np.mean(all_rmse), np.mean(all_R))) 
        log.info('(2) [epoch %d avg all_mae %.5f all_rmse %.5f all_R %.5f]'
                    % (epoch, np.mean(all_mae2), np.mean(all_rmse2), np.mean(all_R2))) 
        # print(f"{np.mean(all_mae)=}")
        # print(f"{np.mean(all_rmse)=}")
        # print(f"{np.mean(all_R)=}")

            # break
        # break
    #  
