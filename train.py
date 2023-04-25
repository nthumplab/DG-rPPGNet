import torch
import torch.nn as nn
import torchvision.transforms as T
from torch import optim


from einops import rearrange


from model import rPPG_estimator, Encoder, Separator, Classifier, Decoder_video, GradReverse, Difficult_Transform, Project_Head
from load_save_model import load_model, save_model

#from IrrelevantPowerRatio import IrrelevantPowerRatio

from util import *
from dataloader import get_loader
from loss import *
from loss.SupConLoss import SupConLoss
from tool_function import calc_mean_std, norm_sigma, Permutation


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    

args = get_args()
trainName, _ = get_name(args)
log = get_logger(f"logger/train/{args.train_dataset}/", trainName)

print(f"{trainName=}")

class_num = 2
domain_order = [1, 0]
medium_ch = 16
id_num_dict = {"U":34, "P":7, "C":28}
id_num = id_num_dict[args.train_dataset[0]] + id_num_dict[args.train_dataset[1]]
print("id_num",id_num)


result_dir = f"./results/{args.train_dataset}/{trainName}/"
os.makedirs(f"{result_dir}/weight", exist_ok=True)

seq_len = args.train_T*args.fps
train_loader = get_loader(_datasets=list(args.train_dataset),
                          _seq_length=seq_len,
                          batch_size=args.bs,
                          train=True,
                          if_bg=False,)

conv_type='vanilla'
estimator_rPPG = rPPG_estimator(in_ch=medium_ch, seq_length=seq_len).to(device).train()
estimator_rPPG_G = rPPG_estimator(in_ch=medium_ch, seq_length=seq_len, conv_type=conv_type).to(device).train()
feature_extractor = Encoder(medium_channels=medium_ch, task="global").to(device).train()
separator_rPPG = Separator(in_ch=medium_ch, out_ch=medium_ch, task="rPPG").to(device).train()
separator_id = Separator(in_ch=medium_ch, out_ch=medium_ch, task="id").to(device).train()
separator_domain = Separator(in_ch=medium_ch, out_ch=medium_ch, task="domain").to(device).train()
classifier_id = Classifier(in_ch=medium_ch, class_num=id_num, seq_length=seq_len, task="id").to(device).train()
classifier_domain = Classifier(in_ch=medium_ch, class_num=class_num, seq_length=seq_len, task="domain").to(device).train()
decoder = Decoder_video(in_ch=medium_ch*3, out_ch=medium_ch).to(device).train()

d_trans = Difficult_Transform([class_num,medium_ch,1,1,1]).to(device).train()
p_head = Project_Head(medium_ch, seq_length=seq_len).to(device).train()

optimizer_estimator_rPPG = optim.Adam(estimator_rPPG.parameters(), lr=args.lr)
optimizer_estimator_rPPG_G = optim.Adam(estimator_rPPG_G.parameters(), lr=args.lr)
optimizer_feature_extractor = optim.Adam(feature_extractor.parameters(), lr=args.lr)
optimizer_separator_rPPG = optim.Adam(separator_rPPG.parameters(), lr=args.lr)
optimizer_separator_id = optim.Adam(separator_id.parameters(), lr=args.lr)
optimizer_separator_domain = optim.Adam(separator_domain.parameters(), lr=args.lr)
optimizer_classifier_id = optim.Adam(classifier_id.parameters(), lr=args.lr)
optimizer_classifier_domain = optim.Adam(classifier_domain.parameters(), lr=args.lr)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=args.lr)

optimizer_d_trans = optim.Adam(d_trans.parameters(), lr=args.lr)
optimizer_p_head = optim.Adam(p_head.parameters(), lr=args.lr)


NP_criterion = NegPearsonLoss().to(device)
recon_criterion = nn.L1Loss().to(device)
cross_entropy_criterion = nn.CrossEntropyLoss().to(device)
con_criterion = SupConLoss(device=device).to(device)
criterion_cosine = nn.CosineSimilarity().to(device)
recon_loss_weight = 100
recon_video_loss_weight = 100


for epoch in range(args.epoch):
    
    print(f"epoch_train: {epoch}/{args.epoch}:")
    for step, (face_frames, image_paths, ppg_label, bg_frames, bg_paths, domain_labels, id_labels) in enumerate(train_loader):
        

        face_frames = rearrange(face_frames, 'b d t c h w -> (b d) t c h w').to(device)
        label_rPPG = rearrange(ppg_label, 'b d t -> (b d) t').to(device)
        label_id = id_labels.squeeze().to(device)
        label_domain = domain_labels.squeeze().to(device)
        label_id = torch.max(label_id, 1)[1]
        label_domain = torch.max(label_domain, 1)[1]
        

        optimizer_estimator_rPPG.zero_grad()
        optimizer_estimator_rPPG_G.zero_grad()
        optimizer_feature_extractor.zero_grad()
        optimizer_separator_rPPG.zero_grad()
        optimizer_separator_id.zero_grad()
        optimizer_separator_domain.zero_grad()
        optimizer_classifier_id.zero_grad()
        optimizer_classifier_domain.zero_grad()
        optimizer_decoder.zero_grad()
        optimizer_p_head.zero_grad()


        # training main network

        global_feat = feature_extractor(face_frames)

        rPPG_feat = separator_rPPG(global_feat)
        id_feat = separator_id(global_feat)
        domain_feat = separator_domain(global_feat)

        recon_feat, recon_video = decoder(rPPG_feat, id_feat, domain_feat)
        
        disent_rPPG = estimator_rPPG(rPPG_feat)[:,0,:,0,0]
        disent_id = classifier_id(id_feat)
        disent_domain = classifier_domain(domain_feat)
        global_rPPG = estimator_rPPG_G(global_feat)[:,0,:,0,0]

        #loss
        disent_rPPG_loss = NP_criterion(disent_rPPG, label_rPPG)
        disent_id_loss = cross_entropy_criterion(disent_id, label_id)
        disent_domain_loss = cross_entropy_criterion(disent_domain, label_domain)
        recon_feat_loss = recon_loss_weight * recon_criterion(recon_feat, global_feat)
        recon_video_loss = recon_video_loss_weight * recon_criterion(recon_video, face_frames)
        global_rPPG_loss = NP_criterion(global_rPPG, label_rPPG)
        part1_loss = disent_rPPG_loss + disent_id_loss + disent_domain_loss + recon_feat_loss + recon_video_loss + global_rPPG_loss
        part1_loss.backward()

        # Permutation

        domain_feat = domain_feat.detach()
        rPPG_feat = rPPG_feat.detach()
        id_feat = id_feat.detach()

        rng_domain_feat = Permutation(domain_feat, permu_order=domain_order)
        permu_domain_label = Permutation(label_domain, permu_order=domain_order)
        permu_feat, permu_video = decoder(rPPG_feat, id_feat, rng_domain_feat)
        
        permu_global_feat = feature_extractor(permu_video)
        permu_rPPG_feat = separator_rPPG(permu_global_feat)
        permu_id_feat = separator_id(permu_global_feat)
        permu_domain_feat = separator_domain(permu_global_feat)

        permu_rPPG = estimator_rPPG(permu_rPPG_feat)[:,0,:,0,0]
        permu_id = classifier_id(permu_id_feat)
        permu_domain = classifier_domain(permu_domain_feat)
        permu_global_rPPG = estimator_rPPG_G(permu_global_feat)[:,0,:,0,0]

        permu_rPPG_feat_loss = recon_criterion(permu_rPPG_feat, rPPG_feat)
        permu_id_feat_loss = recon_criterion(permu_id_feat, id_feat)
        permu_domain_feat_loss = recon_criterion(permu_domain_feat, rng_domain_feat)
        permu_feat_loss = recon_loss_weight * recon_criterion(permu_global_feat, permu_feat)
        part2_loss = permu_rPPG_feat_loss + permu_id_feat_loss + permu_domain_feat_loss + permu_feat_loss

        permu_rPPG_loss = NP_criterion(permu_rPPG, label_rPPG)
        permu_id_loss = cross_entropy_criterion(permu_id, label_id)
        permu_domain_loss = cross_entropy_criterion(permu_domain, permu_domain_label)
        permu_global_rPPG_loss = NP_criterion(permu_global_rPPG, label_rPPG)
        part3_loss = permu_rPPG_loss + permu_id_loss + permu_domain_loss + permu_global_rPPG_loss

        permu_total_loss = part2_loss + part3_loss
        permu_total_loss.backward()

        if(epoch<=300):

            optimizer_estimator_rPPG.step()
            optimizer_estimator_rPPG_G.step()
            optimizer_feature_extractor.step()
            optimizer_separator_rPPG.step()
            optimizer_separator_id.step()
            optimizer_separator_domain.step()
            optimizer_classifier_id.step()
            optimizer_classifier_domain.step()
            optimizer_decoder.step()
            optimizer_p_head.step()

        elif(epoch>300):

            # DG
            domain_feat = domain_feat.detach()
            rPPG_feat = rPPG_feat.detach()
            id_feat = id_feat.detach()
            global_feat = global_feat.detach()

            diff_domain_feat = d_trans(domain_feat)
            dg_feat, dg_video = decoder(rPPG_feat, id_feat, diff_domain_feat)

            dg_global_feat = feature_extractor(dg_video)
            dg_rPPG_feat = separator_rPPG(dg_global_feat)
            dg_id_feat = separator_id(dg_global_feat)
            dg_domain_feat = separator_domain(dg_global_feat)

            z_G = p_head(global_feat)
            z_dg_G = p_head(dg_global_feat)
            zall = torch.cat([z_dg_G.unsqueeze(1), z_G.unsqueeze(1)], dim=1)

            dg_rPPG = estimator_rPPG(dg_rPPG_feat)[:,0,:,0,0]
            dg_id = classifier_id(dg_id_feat)
            dg_global_rPPG = estimator_rPPG_G(dg_global_feat)[:,0,:,0,0]
            
            dg_rPPG_feat_loss =  recon_criterion(dg_rPPG_feat, rPPG_feat)
            dg_id_feat_loss =  recon_criterion(dg_id_feat, id_feat)
            dg_domain_feat_loss =  recon_criterion(dg_domain_feat, diff_domain_feat)
            dg_feat_loss = recon_loss_weight * recon_criterion(dg_global_feat, dg_feat)
            dg_part2_loss = dg_rPPG_feat_loss + dg_id_feat_loss + dg_domain_feat_loss + dg_feat_loss
            
            dg_rPPG_loss = NP_criterion(dg_rPPG, label_rPPG)
            dg_id_loss = cross_entropy_criterion(dg_id, label_id)
            dg_global_rPPG_loss = NP_criterion(dg_global_rPPG, label_rPPG)
            dg_con_loss = con_criterion(zall, adv=False)
            dg_part3_loss = dg_rPPG_loss + dg_id_loss + dg_global_rPPG_loss
            dg_total_loss = dg_part2_loss + dg_part3_loss + dg_con_loss
            dg_total_loss.backward()

            optimizer_estimator_rPPG.step()
            optimizer_estimator_rPPG_G.step()
            optimizer_feature_extractor.step()
            optimizer_separator_rPPG.step()
            optimizer_separator_id.step()
            optimizer_separator_domain.step()
            optimizer_classifier_id.step()
            optimizer_classifier_domain.step()
            optimizer_decoder.step()
            optimizer_p_head.step()


            # updata learnable randomization
            optimizer_d_trans.zero_grad()

            global_feat = feature_extractor(face_frames)
            rPPG_feat = separator_rPPG(global_feat)
            id_feat = separator_id(global_feat)
            domain_feat = separator_domain(global_feat)
            diff_domain_feat = d_trans(domain_feat)
            dg_feat, dg_video = decoder(rPPG_feat, id_feat, diff_domain_feat)
            dg_global_feat = feature_extractor(dg_video)

            dg_rPPG_feat = separator_rPPG(dg_global_feat)
            z_G = p_head(global_feat)
            z_dg_G = p_head(dg_global_feat)
            zall = torch.cat([z_dg_G.unsqueeze(1), z_G.unsqueeze(1)], dim=1)
            dg_rPPG_feat = GradReverse.grad_reverse(dg_rPPG_feat, 1.0)
            dg_update_rPPG = estimator_rPPG(dg_rPPG_feat)[:,0,:,0,0]
            dg_global_feat = GradReverse.grad_reverse(dg_global_feat,1.0)
            dg_global_update_rPPG = estimator_rPPG_G(dg_global_feat)[:,0,:,0,0]

            dg_rPPG_update_loss = NP_criterion(dg_update_rPPG, label_rPPG)
            dg_con_update_loss = con_criterion(zall, adv=True)
            dg_global_rPPG_update_loss = NP_criterion(dg_global_update_rPPG, label_rPPG)
            d_trans_loss = dg_con_update_loss + dg_rPPG_update_loss + dg_global_rPPG_update_loss
            d_trans_loss.backward()

            optimizer_d_trans.step()


        if(epoch<=300):
            log.info('[epoch %d step %d] disent_r %.5f global_r %.5f part1_loss %.5f permu_loss %.5f'
                        % (epoch, step, disent_rPPG_loss.item(), global_rPPG_loss.item(), part1_loss.item(), permu_total_loss.item()
                            ))
        else:

            log.info('[epoch %d step %d] disent_r %.5f global_r %.5f part1_loss %.5f permu_loss %.5f dg_loss %.5f d_trans_loss %.5f'
                        % (epoch, step, disent_rPPG_loss.item(), global_rPPG_loss.item(), part1_loss.item(), permu_total_loss.item(), dg_total_loss.item(), d_trans_loss.item()
                            ))
        # exit()


    per_epoch_to_save = 25

    if(epoch < 350):
        per_epoch_to_save = 5
    else: per_epoch_to_save = 1
    
    if(epoch % per_epoch_to_save==0):
        save_model("feature_extractor", feature_extractor, optimizer_feature_extractor, epoch)
        save_model("separator_rPPG", separator_rPPG, optimizer_separator_rPPG, epoch)
        save_model("separator_id", separator_id, optimizer_separator_id, epoch)
        save_model("separator_domain", separator_domain, optimizer_separator_domain, epoch)
        save_model("estimator_rPPG", estimator_rPPG, optimizer_estimator_rPPG, epoch)
        save_model("decoder", decoder, optimizer_decoder, epoch)
        save_model("estimator_rPPG_G", estimator_rPPG_G, optimizer_estimator_rPPG_G, epoch)
        save_model("p_head", p_head, optimizer_p_head, epoch)

        if(epoch % 100==0):
            save_model("classifier_id", classifier_id, optimizer_classifier_id, epoch)
            save_model("classifier_domain", classifier_domain, optimizer_classifier_domain, epoch)
        

        if(epoch > 300):
            
            save_model("d_trans", d_trans, optimizer_d_trans, epoch)


    

    

