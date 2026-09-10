import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import random
import time
# from transformationer import SIA, SmoothStructuredSIA
from transformationer_test import SmoothStructuredSIA
import math
from models.clip_model.model import LayerNorm

from torchvision.utils import save_image

# from dct import *
# import transferattack
# from transferattack.utils import *

import matplotlib.pyplot as plt

NUM_COPIES=5

def pairwise_distance(x, y):

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)

    return dist_mat

class Attacker():
    def __init__(self, model, img_attacker, txt_attacker, device):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.device = device

    def attack(self, imgs, txts, txt2img, all_txt_supervisions, device='cpu', max_length=30, scales=[0.5,0.75,1.25,1.5], masks=None, **kwargs):
        """
        all_texts 是从 texts 中随机采样的 一种外部语料？
        """

        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
        # # 基于图片监督，为每个文本生成添加扰动后的对抗文本
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
            txt_token_supervisions = txts_output.get('text_embed', None)

        start_time = time.time()
        adv_imgs, last_adv_imgs, average_loss = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, all_txt_supervisions,
                                                                      scales=scales, txt_embeds=txt_supervisions,
                                                                      txt_token_embeds=txt_token_supervisions)
        end_time = time.time()
        execuate_time = end_time - start_time

        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            last_adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(last_adv_imgs))
            last_adv_img_supervisions = last_adv_imgs_outputs['image_feat'][txt2img]

        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
                                                       adv_img_embeds=adv_img_supervisions,
                                                       last_adv_img_embeds=last_adv_img_supervisions)
        return adv_imgs, adv_txts, average_loss, execuate_time

def normalize_importance_map(importance_map, eps=1e-12):
    flat = importance_map.flatten(1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + eps)
    return flat.view_as(importance_map)


def grad_importance_from_avg_grad(avg_grad, eps=1e-12):
    # avg_grad: [B, C, H, W] -> [B, 1, H, W]
    return normalize_importance_map(avg_grad.abs().mean(dim=1, keepdim=True), eps=eps)


def text_guidance_per_image(text_embeds, txt2img, batch_size):
    if text_embeds is None:
        return None

    if text_embeds.dim() == 3:
        text_embeds = text_embeds[:, 0, :]
    elif text_embeds.dim() != 2:
        return None

    if text_embeds.size(0) == batch_size:
        return text_embeds

    if txt2img is None or len(txt2img) != text_embeds.size(0):
        return None

    txt2img = torch.as_tensor(txt2img, device=text_embeds.device, dtype=torch.long)
    valid = (txt2img >= 0) & (txt2img < batch_size)
    if valid.sum().item() == 0:
        return None

    txt2img = txt2img[valid]
    text_embeds = text_embeds[valid]
    per_image = torch.zeros(batch_size, text_embeds.size(-1), device=text_embeds.device, dtype=text_embeds.dtype)
    counts = torch.zeros(batch_size, 1, device=text_embeds.device, dtype=text_embeds.dtype)
    per_image.index_add_(0, txt2img, text_embeds)
    counts.index_add_(0, txt2img, torch.ones(text_embeds.size(0), 1, device=text_embeds.device, dtype=text_embeds.dtype))
    return per_image / counts.clamp_min(1.0)


def token_text_attention_map(image_tokens, text_embed, target_size=None):
    """
    image_tokens: [B, N, D], usually image_embed from ALBEF/TCL.
    text_embed:   [B, D], one text guidance vector per image.
    """
    if image_tokens is None or text_embed is None:
        return None

    if image_tokens.dim() != 3:
        return None

    if text_embed.dim() == 3:
        text_embed = text_embed[:, 0, :]
    elif text_embed.dim() != 2:
        return None

    B, num_tokens, dim = image_tokens.shape
    if text_embed.size(0) != B or text_embed.size(-1) != dim:
        return None

    grid = int(math.sqrt(num_tokens))
    if grid * grid == num_tokens:
        patch_tokens = image_tokens
    else:
        patch_count = num_tokens - 1
        grid = int(math.sqrt(patch_count))
        if grid * grid != patch_count:
            return None
        patch_tokens = image_tokens[:, 1:, :]

    patch_tokens = F.normalize(patch_tokens, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    score = (patch_tokens * text_embed.unsqueeze(1)).sum(dim=-1)
    score = score.view(B, 1, grid, grid)

    if target_size is not None and score.shape[-2:] != target_size:
        score = F.interpolate(score, size=target_size, mode='bilinear', align_corners=False)

    return normalize_importance_map(score)


def build_importance_map(images_embeds, text_embeds, avg_grad, txt2img=None, lam=0.6, blur=None,
                         save_debug=False,
                         debug_path="/home/jyg/workingshops/SA-AET-main/result/SA-AET-RE/attention_m.png"):
    m_grad = grad_importance_from_avg_grad(avg_grad)

    # save_image(m_grad, f"/home/jyg/workingshops/SA-AET-main/result/SA-AET-RE/grad.png")

    text_per_image = text_guidance_per_image(text_embeds, txt2img, m_grad.size(0))
    m_attn = token_text_attention_map(images_embeds, text_per_image, target_size=m_grad.shape[-2:])
    # save_image(text_per_image, f"/home/jyg/workingshops/SA-AET-main/result/SA-AET-RE/text_per_image.png")

    if m_attn is None:
        m = m_grad
    else:
        m = lam * m_grad + (1 - lam) * m_attn

    if blur is not None:
        m = blur(m)

    if save_debug:
        save_image(m, debug_path)
    return normalize_importance_map(m)

class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, device=None, num_block=4,
                soft_mask_min=0.35,
                attn_blur_kernel=11,
                attn_gamma=0.8,
                coarse_importance_blur_kernel=21,
                fine_importance_blur_kernel=9,
                ablation_mode='full',
                save_importance_debug=False):
        if ablation_mode not in ['full', 'crossram_only']:
            raise ValueError("ablation_mode must be 'full' or 'crossram_only'")

        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.device = device
        self.ablation_mode = ablation_mode
        self.save_importance_debug = save_importance_debug
        # self.clip_dtype = torch.float32
        self.warping = SmoothStructuredSIA(num_copies=NUM_COPIES, device=device, num_block=num_block,
                                        guided_ratio=0.5,
                                        coarse_fuse_ratio=0.7,
                                        coarse_importance_blur_kernel=coarse_importance_blur_kernel,
                                        fine_importance_blur_kernel=fine_importance_blur_kernel,
                                        use_importance_map=False,
                                        inactive_scale=0.2)
        self.use_importance_map = ablation_mode == 'crossram_only'
        self.projection_matrix = None
        self.decay = 1

        # guidance related
        self.soft_mask_min = soft_mask_min
        self.attn_gamma = attn_gamma
        self.attn_blur = transforms.GaussianBlur(
            kernel_size=attn_blur_kernel,
            sigma=max(1.0, attn_blur_kernel * 0.15)
        )

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img):
        # U, S, V = torch.svd(adv_imgs_embeds)
        # u1 = U[:, 0].reshape(-1, 1)
        # s1 = S[0]
        # v1 = V[0, :].reshape(1, -1)
        # rank1 = u1 * s1 * v1.T
        # adv_imgs_embeds_r1 = adv_imgs_embeds - rank1
        # alpha = random.randint(1, 100)
        # adv_imgs_embeds = alpha / 100 * adv_imgs_embeds_r1 + (1 - alpha / 100) * adv_imgs_embeds

        adv_imgs_embeds = adv_imgs_embeds @ self.projection_matrix
        txts_embeds = txts_embeds @ self.projection_matrix
        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T

        it_labels = torch.zeros(it_sim_matrix.shape).to(self.device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos

        return loss

    def rand3Num(self): ### num1 -> adv num2-> clean num3->last
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2

            if 1 <= num3 <= 100 and num1 < num3 and num3 < num2:
                break

        return (num1, num2, num3)

    def get_perturbation(self, perturbation, grad):

        # grad = self.low_pass_filter(grad) # 影响精度
        perturbation = self.step_size * torch.sign(grad)
        perturbation = torch.clamp(perturbation, -self.eps, self.eps)
        # delta = clamp(delta, img_min-data, img_max-data)
        return perturbation

    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad / (grad_norm + 1e-12) # 防止除以0
        return momentum * self.decay + grad

    @staticmethod
    def _normalize_map(x, eps=1e-12):
        flat = x.flatten(1)
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        flat = (flat - mn) / (mx - mn + eps)
        return flat.view_as(x)

    def build_soft_mask(self, score_map):
        score_map = score_map.clamp(min=0.0).pow(self.attn_gamma)
        score_map = self._normalize_map(score_map)
        score_map = self.attn_blur(score_map)
        score_map = self._normalize_map(score_map)
        return self.soft_mask_min + (1.0 - self.soft_mask_min) * score_map

    def build_fused_importance_map(self, avg_grad):
        grad_map = self._normalize_map(avg_grad.abs().mean(dim=1, keepdim=True))
        # 若 image_embed 为 [B,D]，此时退化为grad_map

        soft_mask = self.build_soft_mask(grad_map)
        fused = grad_map * soft_mask
        fused = self._normalize_map(fused)
        return fused.detach()

    def compute_avg_grad_and_importance(self, grads):
        # grad 使用的累计梯度

        temp_grad = grads / float(NUM_COPIES)
        avg_grad = temp_grad / (torch.mean(torch.abs(temp_grad), dim=(1,2,3), keepdim=True) + 1e-12)
        importance_map = self.build_fused_importance_map(avg_grad=avg_grad)
        return importance_map.detach()

    def _inference_image(self, model, imgs):
        if self.normalization is not None:
            return model.inference_image(self.normalization(imgs))
        return model.inference_image(imgs)

    def _build_crossram_map(self, model, adv_imgs, avg_grad, txt2img, txt_token_embeds):
        with torch.no_grad():
            importance_output = self._inference_image(model, adv_imgs)
            image_tokens = importance_output.get('image_embed', None)

        return build_importance_map(
            images_embeds=image_tokens,
            text_embeds=txt_token_embeds,
            avg_grad=avg_grad,
            txt2img=txt2img,
            save_debug=self.save_importance_debug
        )

    def _loss_on_identity_copies(self, model, adv_imgs, b, txt_embeds, txt2img):
        identity_imgs = torch.cat([adv_imgs for _ in range(NUM_COPIES)], dim=0)
        adv_imgs_output = self._inference_image(model, identity_imgs)
        adv_imgs_embeds = adv_imgs_output['image_feat']

        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        for i in range(NUM_COPIES):
            loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img)
            loss += loss_item
        return loss

    def _txt_guided_attack_crossram_only(self, model, imgs, txt2img, all_txt_supervisions, txt_embeds=None,
                                         txt_token_embeds=None):
        device = self.device
        model.eval()

        b, _, _, _ = imgs.shape

        perturbation = torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        momentum = torch.zeros_like(imgs).detach().to(device)
        adv_imgs = imgs.detach() + perturbation
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        if self.projection_matrix is None:
            U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32))
            projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t()
            self.projection_matrix = projection_matrix

        last_adv_imgs = None

        average_loss_list = []
        for step in range(self.steps):
            if last_adv_imgs is not None:
                clone_adv_imgs = adv_imgs.clone()

                loss_list = []
                samples = [self.rand3Num() for _ in range(NUM_COPIES)]
                sub_momentum = torch.zeros_like(imgs).detach().to(device)
                importance_grads = torch.zeros_like(imgs).detach().to(device)

                for sample in samples:
                    adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                sample[2] / 100) * last_adv_imgs
                    adv_imgs = adv_imgs.detach().clone().requires_grad_(True)

                    adv_imgs_output = self._inference_image(model, adv_imgs)
                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    del adv_imgs_output

                    model.zero_grad()
                    with torch.enable_grad():
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img)
                    adv_imgs_grad = torch.autograd.grad(loss, adv_imgs, retain_graph=False, create_graph=False)[0]
                    loss_list.append(loss.detach().item())
                    importance_grads += adv_imgs_grad.detach()

                    sub_momentum = self.get_momentum(adv_imgs_grad, sub_momentum)
                    perturbation = self.get_perturbation(perturbation, sub_momentum)

                    adv_imgs = clone_adv_imgs.detach() + perturbation
                    adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

                    adv_imgs_output = self._inference_image(model, adv_imgs)
                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img)
                    loss.backward()

                candidate_index = loss_list.index(max(loss_list))
                adv_imgs = (samples[candidate_index][0] / 100) * clone_adv_imgs + (
                            samples[candidate_index][1] / 100) * imgs + (samples[candidate_index][2] / 100) * last_adv_imgs
                adv_imgs = adv_imgs.detach().clone().requires_grad_(True)
                importance_seed_grad = importance_grads
                update_base = clone_adv_imgs.detach()
            else:
                last_adv_imgs = adv_imgs.clone()
                adv_imgs = adv_imgs.detach().clone().requires_grad_(True)
                importance_seed_grad = None
                update_base = adv_imgs.detach()

            model.zero_grad()
            with torch.enable_grad():
                loss = self._loss_on_identity_copies(model, adv_imgs, b, txt_embeds, txt2img)
            adv_imgs.retain_grad()
            loss.backward()

            average_loss_list.append(loss.detach().item())

            raw_grad = adv_imgs.grad.detach()
            if importance_seed_grad is None:
                importance_seed_grad = raw_grad

            importance_map = self._build_crossram_map(
                model=model,
                adv_imgs=adv_imgs.detach(),
                avg_grad=importance_seed_grad,
                txt2img=txt2img,
                txt_token_embeds=txt_token_embeds
            )
            soft_mask = self.build_soft_mask(importance_map)
            weighted_grad = raw_grad * soft_mask
            del importance_map

            momentum = self.get_momentum(weighted_grad, momentum)
            perturbation = self.get_perturbation(perturbation, momentum)

            adv_imgs = update_base + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            if last_adv_imgs is not None and step > 0:
                last_adv_imgs = update_base.clone()

        average_loss = np.average(average_loss_list)
        return adv_imgs, last_adv_imgs, -average_loss

    def txt_guided_attack(self, model, imgs, txt2img, all_txt_supervisions, scales=None, txt_embeds=None,
                          txt_token_embeds=None):
        if self.ablation_mode == 'crossram_only':
            return self._txt_guided_attack_crossram_only(
                model=model,
                imgs=imgs,
                txt2img=txt2img,
                all_txt_supervisions=all_txt_supervisions,
                txt_embeds=txt_embeds,
                txt_token_embeds=txt_token_embeds
            )

        device = self.device
        model.eval()

        b, _, _, _ = imgs.shape

        perturbation = torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        momentum = torch.zeros_like(imgs).detach().to(device)
        adv_imgs = imgs.detach() + perturbation
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        if self.projection_matrix is None:
            U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32)) # U 256
            projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t() #! core
            self.projection_matrix = projection_matrix

        last_adv_imgs = None

        start_time = time.time()
        # ratio_list = []

        average_loss_list = []
        for step in range(self.steps):  # self.steps=10
            if last_adv_imgs != None: 

                clone_adv_imgs = adv_imgs.clone()

                loss_list = []
                samples = [self.rand3Num() for _ in range(NUM_COPIES)]
                sub_momentum = torch.zeros_like(imgs).detach().to(device)

                # 使用单个张量累计梯度
                importance_grads = torch.zeros_like(imgs).detach().to(device)

                for _, sample in enumerate(samples):
                    adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                sample[2] / 100) * last_adv_imgs
                    adv_imgs = adv_imgs.detach().clone().requires_grad_(True)

                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)

                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    del adv_imgs_output # 释放内存

                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img)  #? 利用外部语料将原始矩阵进行映射
                    adv_imgs.retain_grad()
                    # loss.backward()
                    adv_imgs_grad = torch.autograd.grad(loss, adv_imgs, retain_graph=False, create_graph=False)[0]
                    loss_list.append(loss.detach().item())

                    # grad = adv_imgs.grad
                    # grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                    # perturbation = self.step_size * grad.sign()
                    importance_grads += adv_imgs_grad.detach()

                    sub_momentum = self.get_momentum(adv_imgs_grad, sub_momentum)

                    perturbation = self.get_perturbation(perturbation, sub_momentum)

                    adv_imgs = clone_adv_imgs.detach() + perturbation
                    adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)

                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img) 
                    loss.backward()

                candidate_index = loss_list.index(max(loss_list))
                # ratio_list.append(samples[candidate_index])

                adv_imgs = (samples[candidate_index][0] / 100) * clone_adv_imgs + (
                            samples[candidate_index][1] / 100) * imgs + (samples[candidate_index][2] / 100) * last_adv_imgs
                adv_imgs = adv_imgs.detach().clone().requires_grad_(True)

                # ========== New
                importance_map = None
                if self.use_importance_map:
                    with torch.no_grad():
                        if self.normalization is not None:
                            importance_output = model.inference_image(self.normalization(adv_imgs))
                        else:
                            importance_output = model.inference_image(adv_imgs)
                        image_tokens = importance_output.get('image_embed', None)
                    importance_map = build_importance_map(
                        images_embeds=image_tokens,
                        text_embeds=txt_token_embeds,
                        avg_grad=importance_grads,
                        txt2img=txt2img
                    )
                    #? 使用sia是否会破坏图像的全局结构，导致文本监督过程损失信息，直接应用sia性能一般
                
                # 使用 save_image 保存importance_map
                # save_image(importance_map, f"/home/jyg/workingshops/SA-AET-main/result/SA-AET-RE/importance_map.png")

                scaled_imgs = self.warping.transform(adv_imgs, importance_map=importance_map)
                del importance_map

                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)

                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img)
                        loss += loss_item
                adv_imgs.retain_grad()
                loss.backward()
                # adv_imgs_grad = torch.autograd.grad(loss, adv_imgs, retain_graph=False, create_graph=False)[0]

                average_loss_list.append(loss.detach().item())

                momentum = self.get_momentum(adv_imgs.grad, momentum)
                # momentum = self.get_momentum(adv_imgs_grad, momentum)

                perturbation = self.get_perturbation(perturbation, momentum)

                adv_imgs = clone_adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                last_adv_imgs = clone_adv_imgs.clone()
            else:
                last_adv_imgs = adv_imgs.clone()
                adv_imgs.requires_grad_()
                scaled_imgs = self.warping.transform(adv_imgs)

                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)

                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img)
                        loss += loss_item
                loss.backward()

                average_loss_list.append(loss.detach().item())

                # grad = adv_imgs.grad
                # grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                # perturbation = self.step_size * grad.sign()
                momentum = self.get_momentum(adv_imgs.grad, momentum)

                perturbation = self.get_perturbation(perturbation, momentum)

                adv_imgs = adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        end_time = time.time()

        elapsed_time = end_time - start_time

        # 用于后续绘制 loss curve
        average_loss = np.average(average_loss_list)

        return adv_imgs, last_adv_imgs, -average_loss

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, last_adv_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                if adv_img_embeds == None:
                    loss = self.loss_func(replace_embeds, img_embeds, i)
                else:
                    loss = self.text_ratios[0] * self.loss_func(replace_embeds, img_embeds, i) + self.text_ratios[
                        1] * self.loss_func(replace_embeds, adv_img_embeds, i) + self.text_ratios[2] * self.loss_func(
                        replace_embeds, last_adv_img_embeds, i) # 公式（7）
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse
    
    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
