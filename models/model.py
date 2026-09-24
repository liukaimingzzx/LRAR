import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
from einops import rearrange

from models.attention_layer import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine
from utils.loss import FocalLoss, DiceLoss


class LRAR(nn.Module):
    def __init__(self, args):
        super(LRAR, self).__init__()
        self.backbone_name = args.backbone_name

        # 优先使用本地缓存的仓库和权重，避免网络问题
        local_repo_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
        local_checkpoint_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
        if os.path.exists(local_repo_dir):
            print(f"Loading vision encoder from local repo: {local_repo_dir}")
            self.vision_encoder = torch.hub.load(
                local_repo_dir,
                args.backbone_name,
                source='local',
                pretrained=False,
            )
            if os.path.exists(local_checkpoint_path):
                state_dict = torch.load(local_checkpoint_path, map_location='cpu')
                self.vision_encoder.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from: {local_checkpoint_path}")
        else:
            print("Loading vision encoder from torch.hub (network)...")
            self.vision_encoder = torch.hub.load(
                'facebookresearch/dinov2',
                args.backbone_name,
            )

        self.hidden_dim = 384  # vits14
        self.d_scale = 14
        self.vision_encoder.eval()

        # conv downsample for mask
        self.mask_downsample = nn.Conv2d(
            1, 1,
            kernel_size=self.d_scale,
            stride=self.d_scale,
            padding=1,
            bias=False
        )
        nn.init.constant_(self.mask_downsample.weight, 1.0)

        self.nheads = 4
        self.pre_norm = False
        self.learnable_proxies = nn.Embedding(args.num_learnable_proxies, self.hidden_dim)

        self.rm_ca, self.rm_sa = self.attention_module()   # RM Module
        self.afl_ca, self.afl_sa = self.attention_module() # AFL Module
        self.pe_layer = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)

        # losses
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

        # contra loss hyper-params
        self.contra_temperature = getattr(args, "contra_temperature", 1)
        self.contra_weight = getattr(args, "contra_weight", 1)

        # suppression loss hyper-param nrs模块的权重
        self.supp_weight = getattr(args, "supp_weight", 0.5)

    def attention_module(self):
        cross_attention = CrossAttentionLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dropout=0.0,
            normalize_before=self.pre_norm,
        )
        self_attention = SelfAttentionLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dropout=0.0,
            normalize_before=self.pre_norm,
        )
        return cross_attention, self_attention

    @torch.no_grad()
    def feature_forward(self, image):
        """
        image: (b, num, c, h, w)
        feat:  (b, num, hw, c)
        """
        b, num, c, h, w = image.shape
        feat = self.vision_encoder.get_intermediate_layers(image.view(-1, c, h, w))[0]
        feat = feat.view(b, num, -1, self.hidden_dim)
        return feat

    def get_mask(self, mask, target_size):
        """
        mask: (b, num, h, w)
        return: (b, num, target_size*target_size, 1)
        """
        b, _, _, _ = mask.shape
        mask = rearrange(mask, 'b num h w -> (b num) h w').unsqueeze(1)
        mask = self.mask_downsample(mask)
        mask[mask >= 1] = 1

        if mask.shape[-2] != target_size:
            mask = F.interpolate(mask, size=(target_size, target_size), mode='nearest')

        mask = rearrange(mask, '(b num) 1 p_h p_w -> b num (p_h p_w)', b=b).unsqueeze(-1)
        return mask

    def get_query_mask_flat(self, query_mask, num, target_size):
        """
        query_mask: (b, h, w) or (b, 1, h, w)
        return: (b, num*target_size*target_size, 1)
        """
        if query_mask.dim() == 3:
            query_mask = query_mask.unsqueeze(1)  # (b,1,h,w)

        b = query_mask.shape[0]

        # 复制到每个 query view
        query_mask = query_mask.repeat(1, num, 1, 1)  # (b,num,h,w)
        query_mask = rearrange(query_mask, 'b num h w -> (b num) 1 h w')
        query_mask = F.interpolate(query_mask.float(), size=(target_size, target_size), mode='nearest')
        query_mask = rearrange(query_mask, '(b num) 1 h w -> b (num h w) 1', b=b, num=num)

        # 二值化，避免插值/数据类型带来的小数误差
        query_mask = (query_mask >= 0.5).float()
        return query_mask

    def nn_search(self, query_feat, support_feat, support_mask=None, mode='max'):
        """
        query_feat:   (b, num_q, M, c)
        support_feat: (b, num_s, N, c)
        support_mask: (b, num_s, h, w)
        """
        _, num_q, M, _ = query_feat.shape
        _, num_s, N, _ = support_feat.shape

        query_feat = rearrange(F.normalize(query_feat, dim=-1), 'b num_q M c -> b (num_q M) c')
        support_feat = rearrange(F.normalize(support_feat, dim=-1), 'b num_s N c -> b (num_s N) c')

        if support_mask is not None:
            support_mask_ = self.get_mask(support_mask, int(N ** 0.5))
            support_mask = rearrange(support_mask_, 'b num_s N 1 -> b 1 (num_s N)').repeat(1, M, 1)
        else:
            support_mask_ = None
            support_mask = 1

        similarity_map = (1 + torch.einsum('bmc,bnc->bmn', query_feat, support_feat)) / 2 * support_mask

        if mode == 'max':
            pseudo_mask = similarity_map.max(dim=-1)[0]
        elif mode == 'mean':
            pseudo_mask = similarity_map.mean(dim=-1)
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        pseudo_mask = rearrange(pseudo_mask, 'b (num_q M) -> b num_q M 1', num_q=num_q)

        return pseudo_mask, support_mask_

    def attention_forward(self, cross_layer, self_layer, query_embed, key_feat, value_feat, feat_mask=None):
        """
        query_embed: (num_q, c) or nn.Embedding or (b, num_q, c)
        key_feat:    (b, num_s, M, c)
        value_feat:  (b, num_s, M, c)
        feat_mask:   (b, num_s, M, 1)
        """
        B, _, _, _ = value_feat.shape

        if isinstance(query_embed, nn.Embedding):
            q_supp_out = query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        elif query_embed.dim() == 2:
            q_supp_out = query_embed.unsqueeze(1).repeat(1, B, 1)
        else:
            q_supp_out = query_embed.permute(1, 0, 2)

        key = rearrange(key_feat, 'b num M c -> (num M) b c')
        pos_embedding = self.pe_layer(value_feat, feat_mask)
        value = rearrange(value_feat, 'b num M c -> (num M) b c')

        if feat_mask is not None:
            attn_mask = rearrange(feat_mask.squeeze(-1), 'b num M -> b (num M)')
            attn_mask = attn_mask.unsqueeze(1).repeat(self.nheads, q_supp_out.shape[0], 1)
            attn_mask = -1e9 * (1 - attn_mask)
        else:
            attn_mask = None

        output = cross_layer(
            q_supp_out,
            key,
            value,
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos_embedding,
            query_pos=None
        )
        output = self_layer(
            output,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=None
        )

        return output.permute(1, 0, 2)

    def get_res_feat(self, ori_feat, temp_feat):
        """
        ori_feat:  (b, num, h*w, c)
        temp_feat: (b, num, h*w, c) or (b, 1, k, c)
        """
        b, num, h_w, c = ori_feat.shape
        ori_feat_flat = rearrange(ori_feat, 'b num h_w c -> b (num h_w) c')
        temp_feat_flat = rearrange(temp_feat, 'b num h_w c -> b (num h_w) c')

        sim_map = (1 + torch.einsum('bmc,bnc->bmn', ori_feat_flat, temp_feat_flat)) / 2
        max_idx = sim_map.max(dim=-1)[1]
        most_sim_feat = torch.gather(temp_feat_flat, 1, max_idx.unsqueeze(-1).repeat(1, 1, c))

        res_feat = ori_feat_flat - most_sim_feat
        res_feat = rearrange(res_feat, 'b (num h_w) c -> b num h_w c', b=b, num=num)
        return res_feat

    def proxy_contrastive_loss(self, anomaly_proxies, query_feat, query_mask, temperature=0.1):
        """
        anomaly_proxies: (b, 1, p, c)
        query_feat:      (b, num, hw, c)     [冻结特征，仅作监督参考]
        query_mask:      (b, h, w) or (b,1,h,w)

        目标：
        - anomaly_proxies 靠近 abnormal prototype
        - anomaly_proxies 远离 normal prototype

        只在“当前样本同时存在正常像素和异常像素”时计算 loss，
        避免纯正常样本或异常区域为空时产生无意义监督。
        """
        b, _, p, c = anomaly_proxies.shape
        _, num, hw, _ = query_feat.shape
        l = int(hw ** 0.5)

        feat = rearrange(query_feat, 'b num hw c -> b (num hw) c')
        feat = F.normalize(feat, dim=-1)

        mask_flat = self.get_query_mask_flat(query_mask, num=num, target_size=l)  # (b, num*hw, 1)
        abnormal_mask = mask_flat
        normal_mask = 1.0 - mask_flat

        abnormal_count = abnormal_mask.sum(dim=1)  # (b,1)
        normal_count = normal_mask.sum(dim=1)      # (b,1)

        valid_mask = (abnormal_count.squeeze(-1) > 0) & (normal_count.squeeze(-1) > 0)

        if valid_mask.sum() == 0:
            # 返回一个与图连通的 0，避免训练报错
            return anomaly_proxies.sum() * 0.0

        feat_valid = feat[valid_mask]
        abnormal_mask_valid = abnormal_mask[valid_mask]
        normal_mask_valid = normal_mask[valid_mask]
        proxies_valid = anomaly_proxies[valid_mask]   # (bv,1,p,c)

        abnormal_denom = abnormal_mask_valid.sum(dim=1).clamp(min=1.0)  # (bv,1)
        normal_denom = normal_mask_valid.sum(dim=1).clamp(min=1.0)      # (bv,1)

        abnormal_proto = (feat_valid * abnormal_mask_valid).sum(dim=1) / abnormal_denom
        normal_proto = (feat_valid * normal_mask_valid).sum(dim=1) / normal_denom

        abnormal_proto = F.normalize(abnormal_proto, dim=-1)  # (bv,c)
        normal_proto = F.normalize(normal_proto, dim=-1)      # (bv,c)

        proxies_valid = rearrange(proxies_valid, 'b 1 p c -> b p c')
        proxies_valid = F.normalize(proxies_valid, dim=-1)

        pos_logit = (proxies_valid * abnormal_proto.unsqueeze(1)).sum(dim=-1) / temperature  # (bv,p)
        neg_logit = (proxies_valid * normal_proto.unsqueeze(1)).sum(dim=-1) / temperature    # (bv,p)

        logits = torch.stack([pos_logit, neg_logit], dim=-1)  # (bv,p,2)
        labels = torch.zeros((logits.shape[0], logits.shape[1]), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        return loss

    def normal_relative_suppression_loss(self, s_a, s_n, query_mask):
        """
        只在正常区域约束 s_n 应该大于 s_a。

        s_a:        (b, num, hw) or (b, hw)
        s_n:        (b, num, hw) or (b, hw)
        query_mask: (b, h, w) or (b, 1, h, w)

        loss = mean_normal softplus(s_a - s_n)

        含义：
        - 如果正常区域里 s_a > s_n，说明异常响应压过正常响应，容易误报，损失会变大
        - 如果正常区域里 s_n >= s_a，损失会自然减小
        """
        if s_a.dim() == 2:
            s_a = s_a.unsqueeze(1)
        if s_n.dim() == 2:
            s_n = s_n.unsqueeze(1)

        b, num, hw = s_a.shape
        target_size = int(hw ** 0.5)

        mask_flat = self.get_query_mask_flat(query_mask, num=num, target_size=target_size).squeeze(-1)  # (b, num*hw)
        normal_mask = 1.0 - mask_flat

        s_a_flat = rearrange(s_a, 'b num hw -> b (num hw)')
        s_n_flat = rearrange(s_n, 'b num hw -> b (num hw)')

        loss_map = F.softplus(s_a_flat - s_n_flat) * normal_mask

        normal_count = normal_mask.sum(dim=1)
        valid_mask = (normal_count > 0).float()

        loss_per_sample = loss_map.sum(dim=1) / normal_count.clamp(min=1.0)
        loss = (loss_per_sample * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        return loss

    def prepare_test_image(self, img, transform):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        image_tensor = transform(img)

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width = width - width % self.vision_encoder.patch_size
        cropped_height = height - height % self.vision_encoder.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (
            cropped_height // self.vision_encoder.patch_size,
            cropped_width // self.vision_encoder.patch_size
        )
        return image_tensor, grid_size

    def forward(self, args, query_image, query_mask, query_label, support_normal, support_abnormal, mode='train'):
        query_feat = self.feature_forward(query_image)

        loss_i = 0
        loss_p = 0
        loss_c = 0
        loss_s = 0

        if args.n_shot > 0:
            support_n_image, support_n_mask_ = support_normal
            support_n_feat = self.feature_forward(support_n_image)
            n_pseudo_mask, support_n_mask = self.nn_search(query_feat, support_n_feat, 1 - support_n_mask_)
            s_n = n_pseudo_mask.squeeze(-1)

        if args.a_shot > 0:
            support_a_image, support_a_mask_ = support_abnormal
            support_a_feat = self.feature_forward(support_a_image)
            _, support_a_mask = self.nn_search(query_feat, support_a_feat, support_a_mask_)

            if args.n_shot == 0:
                # only abnormal as reference, obtain normal reference from abnormal image
                support_n_feat = support_a_feat.masked_select((1 - support_a_mask).bool()).view(
                    support_a_feat.shape[0], 1, -1, support_a_feat.shape[-1]
                )
                n_pseudo_mask, support_n_mask = self.nn_search(query_feat, support_a_feat, 1 - support_a_mask_)
                s_n = n_pseudo_mask.squeeze(-1)

            # RM Module forward
            support_res_feat = self.get_res_feat(support_a_feat, support_n_feat)
            residual_proxies = self.attention_forward(
                self.rm_ca, self.rm_sa,
                self.learnable_proxies,
                support_a_feat,
                support_res_feat,
                support_a_mask
            )

            # AFL Module forward
            query_res_feat = self.get_res_feat(query_feat, support_n_feat)
            anomaly_proxies = self.attention_forward(
                self.afl_ca, self.afl_sa,
                residual_proxies,
                query_res_feat,
                query_feat,
                None
            ).unsqueeze(1)  # (b,1,p,c)

            # Contra Loss (PCL)
            if mode == 'train' and getattr(args, 'use_pcl', 1):
                loss_c = self.proxy_contrastive_loss(
                    anomaly_proxies=anomaly_proxies,
                    query_feat=query_feat,
                    query_mask=query_mask,
                    temperature=self.contra_temperature
                )

            a_out, _ = self.nn_search(query_feat, anomaly_proxies, mode='mean')
            s_a = a_out.squeeze(-1)

        if args.n_shot > 0 and args.a_shot == 0:
            s_a = 1 - s_n
        elif args.n_shot == 0 and args.a_shot > 0:
            s_n = 1 - s_a
        else:
            assert args.n_shot > 0 or args.a_shot > 0, 'n_shot and a_shot should not be both 0'

        pixel_level_logits = torch.cat([s_n, s_a], dim=1)  # (b, 2, h*w)
        a_score = (s_a + (1 - s_n)) / 2

        if mode == 'train':
            # 正常区域相对抑制模块 (NRS)
            if getattr(args, 'use_nrs', 1):
                loss_s = self.normal_relative_suppression_loss(s_a, s_n, query_mask)

            a_score_topk = torch.topk(a_score, 20, dim=-1)[0].mean(dim=-1)
            image_level_logits = torch.cat([1 - a_score_topk, a_score_topk], dim=-1)

            # Image Level
            loss_i += self.cross_entropy_loss(image_level_logits, query_label.long())

            # Pixel Level
            l = int(pixel_level_logits.shape[-1] ** 0.5)
            pixel_level_logits = rearrange(pixel_level_logits, 'b n (h w) -> b n h w', h=l)
            pixel_level_logits = F.interpolate(
                pixel_level_logits,
                size=query_mask.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            query_mask_n = torch.stack([1 - query_mask, query_mask], dim=1)
            loss_p += self.focal_loss(pixel_level_logits, query_mask_n)
            loss_p += self.dice_loss(pixel_level_logits, query_mask_n)

            # 加上 contra loss 和 suppression loss
            loss_p += self.contra_weight * loss_c
            loss_p += self.supp_weight * loss_s

            return image_level_logits, pixel_level_logits, loss_i, loss_p

        elif mode == 'test':
            return a_score