# EVAL TRANS TODO:

import os

import clip
import numpy as np
import torch
from scipy import linalg

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric

# added
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import io
from tqdm import tqdm

from utils.metrics import *
from utils.misc import T2M_ID2JOINTNAME

from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import kaleido
from dataset.dataset_TM_eval import get_subsamples_loader as get_subsamples_loader_baseline
from dataset.dataset_RTM_eval import get_subsamples_loader as get_subsamples_loader_rptc

def tensorboard_add_image_tsne(writer, codebook, tag, nb_iter, title="T-sne of Pose Codebook", mode = None):
    
    from utils.codebook import cat_id_to_full_group_name

    x = codebook.cpu().detach()
    x_np = x.numpy()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    x_tsne = tsne.fit_transform(x_np)

    label_map = cat_id_to_full_group_name # {i: f'Group {i % 12}' for i in range(len(x_tsne))}
    labels = [label_map[i] for i in range(len(x_tsne))]

    x_tsne_mean = np.mean(x_tsne, axis=0)

    distance_from_mean = np.linalg.norm(x_tsne - x_tsne_mean, axis=1)
    mask = distance_from_mean < 3 * np.std(distance_from_mean)

    x_tsne_filtered = x_tsne[mask]
    labels_filtered = np.array(labels)[mask]

    x_tsne = x_tsne_filtered
    labels = labels_filtered

    df = pd.DataFrame({
        "Dim1": x_tsne[:, 0],
        "Dim2": x_tsne[:, 1],
        "Group": labels
    })

    num_groups = len(df["Group"].unique())
    colors = plt.cm.gist_ncar(np.linspace(0, 1, num_groups))
    markers = ['o', 's', '^', 'v', '<', '>', '1', '2', '3', '4', '8', 'p', '*', 'h', 'H', 'x', 'D', 'd', '|', '_']

    group_names = sorted(df["Group"].unique())
    group_to_color = {group: colors[i % len(colors)] for i, group in enumerate(group_names)}
    group_to_marker = {group: markers[i % len(markers)] for i, group in enumerate(group_names)}

    plt.figure(figsize=(10, 8))
    for group in group_names:
        idx = df["Group"] == group
        plt.scatter(df.loc[idx, "Dim1"], df.loc[idx, "Dim2"],
                    color=group_to_color[group],
                    marker=group_to_marker[group],
                    label=group,
                    edgecolors='k',
                    s=70)

    plt.title("t-SNE Scatter Plot with Group Labels")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    # plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.legend(
        title="Group",
        bbox_to_anchor=(0.5, -0.15),  
        loc='upper center',
        ncol=6,  
        fontsize='small',
        title_fontsize='medium',
        frameon=False
    )

    # writer.add_figure('t-SNE Scatter Plot', plt.gcf())
    # writer.close()

    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=250)
    plt.close()
    buf.seek(0)

    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # (HWC) → (CHW)

    writer.add_image(tag, image_tensor, global_step=nb_iter)


def tensorboard_add_image_pw_sim(writer, codebook, tag, nb_iter, title="Pairwise Similarity Heatmap of Pose Codebook"):
    
    pw_sim = calculate_pairwise_similarity(codebook)
    heatmap = pw_sim.cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(heatmap, cmap='viridis', origin='upper')
    ax.set_title(title, fontsize=14)
    fig.colorbar(cax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=250)
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # (HWC) → (CHW)

    writer.add_image(tag, image_tensor, global_step=nb_iter)


def tensorboard_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None, footer_text=None, footer_fontsize=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    if footer_text is not None:
        plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname, footer_text, footer_fontsize)
    else:
        plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

@torch.no_grad()        
def evaluation_enc(out_dir, val_loader, dec, enc, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    enc.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, code_indices, _ = batch
        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)
            code_indices = enc.k_hot(enc(motion[i:i+1, :m_length[i], :]))
            pred_pose = dec(code_indices)
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length) 

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                
                tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : enc.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : enc.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : enc.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : enc.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : enc.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    enc.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_dec(args, out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, eval_wrapper, num_joints, split, max_motion_len=None, draw = True, save = True, savegif=False, savenpy=False, unit_length=4, is_test=False, eval_loss_list=None, align_root=True, drop_out_residual_quantization=False): 
    net.eval()

    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0
    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    mpjpe = 0.
    pampjpe = 0.
    accel = 0.

    jpe = torch.zeros(num_joints)

    count = 0

    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _, _, code_indices, _, _ = batch
        motion = motion.cuda()
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)

        bs, seq = motion.shape[0], motion.shape[1]

        count += bs

        num_joints = 21 if motion.shape[-1] == 251 else 22
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            # ground truth
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)
            pred_pose, *_ = net(code_indices[i:i+1,:m_length[i]:unit_length].cuda().float(), motion[i:i+1,:m_length[i]].cuda().float(), detach_p_latent=args.detach_p_latent, drop_out_residual_quantization=drop_out_residual_quantization)

            # prediction
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            joint_metrics = compute_joint_metrics(pose_xyz, pred_xyz, m_lengths = [m_length[i]], jointstype="humanml3d", align_root=align_root)

            mpjpe += joint_metrics['mpjpe']
            pampjpe += joint_metrics['pampjpe']
            accel += joint_metrics['accel']
            jpe += joint_metrics['jpe'].detach().cpu() # vector tensor

            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])
        
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        # R-Precision
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    met_mpjpe = mpjpe / count
    met_pampjpe = pampjpe / count
    met_accel = accel / count
    met_jpe = jpe / count # 

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)
    
    # Diversity
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    # R-Precision
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    # Matching score
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, MPJPE:{met_mpjpe}, PA-MPJPE:{met_pampjpe}, ACCEL:{met_accel}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar(f'./{split}/FID', fid, nb_iter)
        writer.add_scalar(f'./{split}/Diversity', diversity, nb_iter)
       
        writer.add_scalar(f'./{split}/top1', R_precision[0], nb_iter)
        writer.add_scalar(f'./{split}/top2', R_precision[1], nb_iter)
        writer.add_scalar(f'./{split}/top3', R_precision[2], nb_iter)
        writer.add_scalar(f'./{split}/matching_score', matching_score_pred, nb_iter)

        # joint feature error
        writer.add_scalar(f'./{split}/MPJPE', met_mpjpe, nb_iter)
        writer.add_scalar(f'./{split}/PA-MPJPE', met_pampjpe, nb_iter)
        writer.add_scalar(f'./{split}/ACCEL', met_accel, nb_iter)

        for id, jpe in enumerate(met_jpe.tolist()):
            joint_name = T2M_ID2JOINTNAME[id]
            writer.add_scalar(f'./{split}/JPE/{joint_name}', jpe, nb_iter)

        if nb_iter % 5000 == 0: 
            for ii in range(4):
                tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        
        if nb_iter % 5000 == 0:
            if eval_loss_list is not None:

                assert len(eval_loss_list) == len(draw_pred) 
                
                for ii in range(4):
                    loss_print = eval_loss_list[ii]
                    footer = f'Loss:{loss_print}'
                    tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None, footer_text=footer, footer_fontsize=25)   
            else: 
                for ii in range(4):
                    tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   
                
            
    
    if fid < best_fid: 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div): 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1: 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2: 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3: 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching: 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_matching.pth'))

    if met_mpjpe < best_mpjpe: 
        msg = f"--> --> \t MPJPE Improved from {best_mpjpe:.5f} to {met_mpjpe:.5f} !!!"
        logger.info(msg)
        best_mpjpe = met_mpjpe
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_mpjpe.pth'))

    if met_pampjpe < best_pampjpe: 
        msg = f"--> --> \t PA-MPJPE Improved from {best_pampjpe:.5f} to {met_pampjpe:.5f} !!!"
        logger.info(msg)
        best_pampjpe = met_pampjpe
        if save:
            torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_pampjpe.pth'))

    if met_accel < best_accel: 
        msg = f"--> --> \t ACCEL Improved from {best_accel:.5f} to {met_accel:.5f} !!!"
        logger.info(msg)
        best_accel = met_accel
        # if save:
        #     torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_accel.pth'))

    if save:
        torch.save({'net': net.state_dict(), 'nb_iter': nb_iter}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, best_pampjpe, best_accel, writer, logger

@torch.no_grad()        
def evaluation_transformer(args, out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, text_encoder, eval_wrapper, optimizer, scheduler, draw = True, save = True, savegif=False, unit_length = 4): 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    val_acc = 0
    val_loss = 0
    # offset = 11

    nb_sample = 0
    nb_total_pred = 0

    for i in range(1):
        for batch in val_loader:
            # validation batch get
            word_embeddings, pos_one_hots, text_data, sent_len, pose, m_length, token, name, indices, keyword_embeddings, *rest = batch

            # shape 
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            # text tokenize
            text = clip.tokenize(text_data, truncate=True).cuda()

            # text embedding
            text_feat = text_encoder.encode_text(text).float().unsqueeze(1) #bs x 1 x 512

            # fine grained text embedding + sentence embedding
            if args.use_keywords:
                text_feat = torch.cat((text_feat, keyword_embeddings.float().cuda()), dim = 1) # bs x 11+1 x 512

            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).float()
            
            for k in range(bs):        
                index_motion = trans.sample(text_feat[k:k+1], False, m_length[k]//unit_length) # 1 x t x code_num -> k-hot vector (bs, seq_len, 394)
                pred_pose, *_ = net.forward(index_motion[:,:,:-2].float(), drop_out_residual_quantization=True) 

                cur_len = pred_pose.shape[1]
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                org_pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, org_pose, m_length) #m_length
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        if draw:
            sub_loader = get_subsamples_loader_baseline(val_loader, num_samples=4, seed=args.seed)

            for sub_batch in sub_loader:
                word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, *rest = sub_batch
                bs, seq = pose.shape[:2]
                num_joints = 21 if pose.shape[-1] == 251 else 22
                   
                text = clip.tokenize(clip_text, truncate=True).cuda()
                text_feat = text_encoder.encode_text(text).float().unsqueeze(1) #bs x 1 x 512
                if args.use_keywords:
                    text_feat = torch.cat((text_feat, keyword_embeddings.float().cuda()), dim = 1) # bs x 11+1 x 512

                
                bs, seq = pose.shape[:2]
                num_joints = 21 if pose.shape[-1] == 251 else 22

                for j in range(bs):

                    index_motion = trans.sample(text_feat[j:j+1], False, m_length[j]//unit_length) # 1 x t x code_num -> k-hot vector (bs, seq_len, 394)                   
                    pred_pose, *_ = net.forward(index_motion[:,:,:-2].float(), drop_out_residual_quantization=True) # (1, T, Jx3)

                    org_pose = val_loader.dataset.inv_transform(pose[j:j+1].detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(org_pose).float().cuda(), num_joints)
                    
                    draw_org.append(pose_xyz[:,:m_length[j]])
                    draw_text.append(clip_text[j])
                        
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    draw_pred.append(pred_xyz)
                    draw_text_pred.append(clip_text[j])

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    writer.add_scalar('./Validation/FID', fid, nb_iter)
    writer.add_scalar('./Validation/Diversity', diversity, nb_iter)
    writer.add_scalar('./Validation/top1', R_precision[0], nb_iter)
    writer.add_scalar('./Validation/top2', R_precision[1], nb_iter)
    writer.add_scalar('./Validation/top3', R_precision[2], nb_iter)
    writer.add_scalar('./Validation/matching_score', matching_score_pred, nb_iter)
    
    if draw:
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans': trans.state_dict(),
                        'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching: 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'trans': trans.state_dict(),
                        'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_matching.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'trans': trans.state_dict(),
                        'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
        
    if save:
        torch.save({'trans': trans.state_dict(),
                    'nb_iter': nb_iter}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss
    

# net: motion decoder
@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, text_encoder, eval_wrapper, 
                                draw = True, save = True, savegif=False, savenpy=False, unit_length = 4, 
                                mm_mode=False, text_encoding_method='baseline',
                                block_size=62, use_keywords=True): 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    p = 0 

    nb_sample = 0
    i = 0

    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, *rest = batch
        bs, seq = pose.shape[:2] # bs: nb sample, seq: sequence length

        num_joints = 21 if pose.shape[-1] == 251 else 22

        # text tokenize
        text = clip.tokenize(clip_text, truncate=True).cuda()
        # text embedding
        feat_clip_text = text_encoder.encode_text(text).float().unsqueeze(1) #bs x 1 x 512
        if use_keywords:
            feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim = 1) # bs x 11+1 x 512

        if mm_mode:
            repeat_num = 30
        else:
            repeat_num = 1

        motion_multimodality_batch = []
        for i in range(repeat_num):
                
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs): 
                index_motion = trans.sample(feat_clip_text[k:k+1], True, m_length[k]//unit_length) # 1 x t x code_num -> token sequence                
                pred_pose, *_ = net.forward(index_motion[:,:,:-2].float(), drop_out_residual_quantization=True) # (1, T, Jx3)
                cur_len = pred_pose.shape[1] 
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])  

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 10)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 10)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if mm_mode:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 5)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    else:
        multimodality = None
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        calculate_multimodality(motion_multimodality, 1)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. (N/A)"
    
    logger.info(msg)
    
    if draw:
        for ii in range(len(draw_org)):
            tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

@torch.no_grad()        
def evaluation_transformer_test_fast(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, text_encoder, eval_wrapper, 
                                draw = True, save = True, savegif=False, savenpy=False, unit_length = 4, 
                                mm_mode=False, text_encoding_method='baseline',
                                block_size=62, max_token_len=49, end_token_id=392, use_keywords=True): 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    p = 0 

    nb_sample = 0
    i = 0

    for batch in val_loader: # 32min x 

        # if text_encoding_method == 'graph_reasoning':
        #     word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, m_tokens_new, V, entities, relations = batch
        # else:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, *rest = batch
        bs, seq = pose.shape[:2] # bs: nb sample, seq: sequence length

        num_joints = 21 if pose.shape[-1] == 251 else 22

        # text tokenize
        text = clip.tokenize(clip_text, truncate=True).cuda()
        # text embedding
        feat_clip_text = text_encoder.encode_text(text).float().unsqueeze(1) #bs x 1 x 512
        # fine grained text embedding + sentence embedding
        if use_keywords:
            feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim = 1) # bs x 11+1 x 512

        if mm_mode:
            repeat_num = 30
        else:
            repeat_num = 1

        motion_multimodality_batch = []
        for i in range(repeat_num):
                
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            ######################################################################################################################################################

            index_motions = trans.sample_fast(feat_clip_text, True) 
            is_end_token = (index_motions[:, :, end_token_id] == 1) # end_token_id = 392
            has_end_token = torch.any(is_end_token, dim=1)

            pred_mo_lens = torch.argmax(is_end_token.int(), dim=1)
            pred_mo_lens[~has_end_token] = max_token_len
            pred_mo_lens = pred_mo_lens.cpu().numpy()

            for k in range(bs):
                index_motion = index_motions[k:k+1] # 1 x t x code_num -> token sequence
                pred_mo_len = pred_mo_lens[k]
                pred_pose, *_ = net.forward(index_motion[:,:pred_mo_len,:-2].float(), drop_out_residual_quantization=True) # (1, T, Jx3)
            
                cur_len = pred_pose.shape[1] 
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])
            
            ######################################################################################################################################################

            

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 10)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 10)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if mm_mode:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 5)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    else:
        multimodality = None
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        calculate_multimodality(motion_multimodality, 1)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. (N/A)"
    
    logger.info(msg)
    
    if draw:
        for ii in range(len(draw_org)):
            tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger


def euclidean_distance_matrix(matrix1, matrix2): 
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1) 
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    
    dist_mat = euclidean_distance_matrix(embedding1, embedding2) # word embedding, motion embedding -> distance calculation
    
    matching_score = dist_mat.trace() 
    
    argmax = np.argsort(dist_mat, axis=1) 

    top_k_mat = calculate_top_k(argmax, top_k) 

    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] >= multimodality_times
    num_per_sent = activation.shape[1]
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_feature_distance(feature_list1, feature_list2):
    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist



def calculate_pairwise_similarity(codebook):

    mean_vector = torch.mean(codebook,dim=0).unsqueeze(0)
    origin_vector = mean_vector

    code_vectors = (codebook - origin_vector)

    norms = torch.norm(code_vectors, dim=1, keepdim=True)

    unit_vecs = code_vectors / norms 

    pw_dir_sim = unit_vecs @ unit_vecs.T 

    return pw_dir_sim




@torch.no_grad()        
def evaluation_residual_transformer(args, out_dir, val_loader, net, trans, r_trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, optimizer, scheduler, draw = True, save = True, savegif=False, unit_length = 4, num_keywords = 11): 
    
    r_trans.eval()

    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    val_acc = 0
    val_loss = 0

    nb_sample = 0
    nb_total_pred = 0

    for i in range(1):
        for batch in val_loader:
            # validation batch get

            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, m_tokens_new, m_res_tokens_new = batch

            # shape 
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            # text tokenize
            text = clip.tokenize(clip_text, truncate=True).cuda()
            
            # text embedding
            feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1) #bs x 1 x 512
            
            if args.use_keywords:
                feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim=1)  # bs x 12 x 512
            
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).float()
            
            # batch index k
            for k in range(bs):
                # prediction --> outputs token sequences
                
                pred_p_codes, pred_r_codes = r_trans.sample(feat_clip_text[k:k+1], trans, offset=num_keywords) # 1 x t x code_num -> k-hot vector (bs, seq_len, 394)
                pred_p_codes = pred_p_codes[:,:,:-2] 

                pred_pose = net.inference(pred_r_codes.float(), pred_p_codes.float()) # (1, T, Jx3) 

                # predicted motion frame length
                cur_len = pred_pose.shape[1]

                # fit to minimum
                pred_len[k] = min(cur_len, seq)

                # crop to size of real frame length
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:

                org_pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, org_pose, m_length) #m_length
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs


        if draw:
            sub_loader = get_subsamples_loader_rptc(val_loader, num_samples=4, seed=args.seed)

            for sub_batch in sub_loader:
            
                word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, m_tokens_new, m_res_tokens_new = sub_batch
            
                bs, seq = pose.shape[:2]
                num_joints = 21 if pose.shape[-1] == 251 else 22
                
                # text tokenize
                text = clip.tokenize(clip_text, truncate=True).cuda()
                feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1) # bs x 1 x 512

                if args.use_keywords:
                    feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim=1)  # bs x 12 x 512
                
                bs, seq = pose.shape[:2]
                num_joints = 21 if pose.shape[-1] == 251 else 22

                for j in range(bs):
    
                    pred_p_codes, pred_r_codes = r_trans.sample(feat_clip_text[j:j+1], trans, offset=num_keywords)
                    pred_p_codes = pred_p_codes[:,:,:-2]
                    pred_pose = net.inference(pred_r_codes.float(), pred_p_codes.float())

                    org_pose = val_loader.dataset.inv_transform(pose[j:j+1].detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(org_pose).float().cuda(), num_joints)
                    
                    draw_org.append(pose_xyz[:,:m_length[j]])
                    draw_text.append(clip_text[j])
                        
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    draw_pred.append(pred_xyz)
                    draw_text_pred.append(clip_text[j])

    
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()

    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    
    # R-Precision scaling
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    # Matching Score scaling
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    writer.add_scalar('./Validation/FID', fid, nb_iter)
    writer.add_scalar('./Validation/Diversity', diversity, nb_iter)
    writer.add_scalar('./Validation/top1', R_precision[0], nb_iter)
    writer.add_scalar('./Validation/top2', R_precision[1], nb_iter)
    writer.add_scalar('./Validation/top3', R_precision[2], nb_iter)
    writer.add_scalar('./Validation/matching_score', matching_score_pred, nb_iter)

    if draw:
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)
    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'r_trans': r_trans.state_dict(),
                        'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching: 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'r_trans': r_trans.state_dict(),
                        'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_matching.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'r_trans': r_trans.state_dict(),
                        'nb_iter': nb_iter}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'r_trans': r_trans.state_dict(),
                    'nb_iter': nb_iter}, os.path.join(out_dir, 'net_last.pth'))

    r_trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, val_acc, val_loss


# net: motion decoder
@torch.no_grad()        
def evaluation_residual_transformer_test(out_dir, val_loader, net, trans, r_trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, unit_length = 4, mm_mode=False, use_keywords=False): 
    
    r_trans.eval()
    trans.eval()
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    import time

    nb_sample = 0
    n_max_save = 20
    saved = 0
    gt_saved = 0

    if mm_mode:
        repeat_num = 30
    else:
        repeat_num = 1

    for batch in val_loader: 

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, *_ = batch
        bs, seq = pose.shape[:2] # bs: nb sample, seq: sequence length

        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1) 
        
        if use_keywords:
            feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim = 1) # bs x 11+1 x 512
            offset = 11
        else:
            offset = 0

        
        motion_multimodality_batch = []
        for i in range(repeat_num):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):  
                pred_p_codes, pred_r_codes = r_trans.sample(feat_clip_text[k:k+1], trans, if_categorial=True, if_residual_categorical=True, offset=offset) # 1 x t x code_num -> token sequence
                pred_p_codes = pred_p_codes[:,:,:-2]

                pred_pose = net.inference(pred_r_codes.float(), pred_p_codes.float()) # (1, T, Jx3) -> Decoder(token seq -> motion)

                cur_len = pred_pose.shape[1] 
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if (draw or savenpy) and saved < n_max_save:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])
                        draw_name.append(name[k])
                    
                    saved += 1
                        
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if (draw or savenpy) and gt_saved < n_max_save:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(n_max_save):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(n_max_save):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])
                    
                    gt_saved += n_max_save
            
                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match
                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)
    
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 10)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 10)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if mm_mode:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 5)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    else:
        multimodality = None
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        calculate_multimodality(motion_multimodality, 1)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. (N/A)"

    logger.info(msg)

    assert len(draw_org) == len(draw_pred) <= n_max_save, \
    f"GT/Pred mismatch: {len(draw_org)} vs {len(draw_pred)} (limit {n_max_save})"
    
    if draw:
        for ii in range(len(draw_org)):
            tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

@torch.no_grad()        
def evaluation_residual_transformer_test_fast(out_dir, val_loader, net, trans, r_trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, unit_length = 4, mm_mode=False, use_keywords=False) : 
    
    r_trans.eval()
    trans.eval()
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    import time

    nb_sample = 0
    n_max_save = 20
    saved = 0
    gt_saved = 0

    if mm_mode:
        repeat_num = 30
    else:
        repeat_num = 1

    for batch in val_loader: # 32min x 

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name, indices, keyword_embeddings, *_ = batch
        bs, seq = pose.shape[:2] # bs: nb sample, seq: sequence length

        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()
        feat_clip_text = clip_model.encode_text(text).float().unsqueeze(1) 

        if use_keywords:
            feat_clip_text = torch.cat((feat_clip_text, keyword_embeddings.float().cuda()), dim = 1) # bs x 11+1 x 512    
            offset = 11
        else:
            offset = 0

        
        motion_multimodality_batch = []
        for i in range(repeat_num):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            pred_p_codes, pred_r_codes, pred_mo_lens = r_trans.sample_fast(feat_clip_text, trans, if_categorial=True, if_residual_categorical=True, offset=offset) # 1 x t x code_num -> token sequence
            pred_p_codes = pred_p_codes[:,:,:-2]

            for k in range(bs): 
                pred_p_code = pred_p_codes[k:k+1] # 1 x t x code_num -> token sequence
                pred_r_code = pred_r_codes[k:k+1]
                pred_mo_len = pred_mo_lens[k]
                pred_pose = net.inference(pred_r_code[:, :pred_mo_len, :].float(), pred_p_code[:, :pred_mo_len, :].float()) # (1, T, Jx3) -> Decoder(token seq -> motion)

                cur_len = pred_pose.shape[1] 
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if (draw or savenpy) and saved < n_max_save:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])
                        draw_name.append(name[k])
                    
                    saved += 1
                        
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if (draw or savenpy) and gt_saved < n_max_save:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(n_max_save):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(n_max_save):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])
                    
                    gt_saved += n_max_save
            
                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match
                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)
    
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 10)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 10)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if mm_mode:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 5)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    else:
        multimodality = None
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        calculate_multimodality(motion_multimodality, 1)
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. (N/A)"

    logger.info(msg)

    assert len(draw_org) == len(draw_pred) <= n_max_save, \
    f"GT/Pred mismatch: {len(draw_org)} vs {len(draw_pred)} (limit {n_max_save})"
    
    if draw:
        for ii in range(len(draw_org)):
            tensorboard_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorboard_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

