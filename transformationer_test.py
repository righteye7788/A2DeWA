import torch
import torch.nn as nn
import numpy as np
from utils import *
import torch.nn.functional as F
from torchvision import transforms
import torch_dct as dct
import scipy.stats as st

import math

# 添加图像块可视化逻辑
import os
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

img_max, img_min = 1., 0

class SmoothStructuredSIA(nn.Module):
    def __init__(
        self,
        image_size=384,
        num_block=3,
        max_disp=0.1,
        max_rotate=5,
        max_scale=0.1,
        num_copies=5,
        device=None,
        importance_mode="soft",      # "none" / "soft" / "topk"
        k_ratio=0.5,
        score_mode="hybrid",         # "mean" / "topq" / "hybrid"
        topq=0.1,
        use_importance_map=True,
        guided_ratio=0.5, # 采样引导比例
        activation_bias=0.3,
        random_aug_ratio=1.15,
        guided_aug_ratio=0.85,
        inactive_scale=0.2,
        coarse_fuse_ratio=0.7,
        coarse_importance_blur_kernel=None,
        fine_importance_blur_kernel=None,
    ):
        super(SmoothStructuredSIA, self).__init__()
        self.image_size = image_size
        self.max_disp = max_disp
        self.max_theta = max_rotate * (math.pi / 180)
        self.max_scale = max_scale
        self.device = device
        self.num_copies = num_copies
        self.num_block = num_block

        self.importance_mode = importance_mode
        self.k_ratio = k_ratio
        self.score_mode = score_mode
        self.topq = topq

        self.use_importance_map = use_importance_map
        self.guided_ratio = guided_ratio
        self.activation_bias = activation_bias
        self.random_aug_ratio = random_aug_ratio
        self.guided_aug_ratio = guided_aug_ratio

        # multi-levels
        self.coarse_fuse_ratio = coarse_fuse_ratio
        self.inactive_scale = inactive_scale

        self.coarse_importance_blur_kernel = coarse_importance_blur_kernel
        self.fine_importance_blur_kernel = fine_importance_blur_kernel

        self.coarse_importance_blur = transforms.GaussianBlur(
            kernel_size=self.coarse_importance_blur_kernel,
            sigma=(max(1.0, self.coarse_importance_blur_kernel * 0.2), max(1.0, self.coarse_importance_blur_kernel * 0.2))
        )
        self.fine_importance_blur = transforms.GaussianBlur(
            kernel_size=self.fine_importance_blur_kernel,
            sigma=(max(1.0, self.fine_importance_blur_kernel * 0.2), max(1.0, self.fine_importance_blur_kernel * 0.2))
        )

    @staticmethod
    def importance_from_grad(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """ 生成空间重要性图
        grad: [B,C,H,W] -> raw importance_map: [B,1,H,W]
        """
        return grad.abs().mean(dim=1, keepdim=True) + eps

    @staticmethod
    def _normalize_map(imp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        B = imp.shape[0]
        flat = imp.reshape(B, -1)
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        flat = (flat - mn) / (mx - mn + eps)
        return flat.view_as(imp)

    @staticmethod
    def _topq_mean(block_imp: torch.Tensor, q: float = 0.1) -> torch.Tensor:
        B = block_imp.shape[0]
        flat = block_imp.reshape(B, -1)
        k = max(1, int(math.ceil(flat.shape[1] * q)))
        topk = torch.topk(flat, k=k, dim=1, largest=True).values
        return topk.mean(dim=1)

    def _compute_block_scores(self, importance_map, coords, H, W):
        if importance_map is None:
            return None, None

        if importance_map.shape[-2:] != (H, W):
            importance_map = F.interpolate(
                importance_map, size=(H, W), mode="bilinear", align_corners=False
            )

        score_list = []
        for (x0, x1, y0, y1) in coords:
            blk = importance_map[:, :, x0:x1, y0:y1]
            mean_score = blk.mean(dim=(1, 2, 3))
            topq_score = self._topq_mean(blk, q=self.topq)

            if self.score_mode == "mean":
                s = mean_score
            elif self.score_mode == "topq":
                s = topq_score
            elif self.score_mode == "hybrid":
                s = 0.5 * mean_score + 0.5 * topq_score
            else:
                raise ValueError(f"Unknown score_mode: {self.score_mode}")
            score_list.append(s)

        scores = torch.stack(score_list, dim=1)
        return scores, importance_map

    def _sample_activation_mask(self, scores: torch.Tensor, mode: str):
        """
        根据块分数采样哪些块被激活。
        random 模式保持全局分布扩展。
        guided 模式让高重要块更容易被激活，但低重要块依然保留参与概率。 得分高的块更容易被激活，而不是100%
        """
        B, nb = scores.shape
        device = scores.device

        if mode == 'random':
            base = torch.full((B, nb), 0.5, device=device)
        elif mode == 'guided':
            norm_scores = scores / (scores.mean(dim=1, keepdim=True) + 1e-12)
            base = 0.5 + self.activation_bias * (norm_scores - 1.0)
        else:
            raise ValueError(f'Unknown transform mode: {mode}')

        base = base.clamp(0.25, 0.85)

        if self.importance_mode == 'topk' and mode == 'guided':
            k = max(1, int(math.ceil(nb * self.k_ratio)))
            topk_idx = torch.topk(scores, k=k, dim=1, largest=True).indices
            mask = torch.zeros(B, nb, device=device)
            mask.scatter_(1, topk_idx, 1.0)
            random_keep = (torch.rand(B, nb, device=device) < 0.15).float()
            mask = torch.maximum(mask, random_keep)
        else:
            mask = (torch.rand(B, nb, device=device) < base).float()

        zero_rows = mask.sum(dim=1) == 0
        if zero_rows.any():
            idx = torch.argmax(scores[zero_rows], dim=1)
            mask[zero_rows] = 0.0
            mask[zero_rows, idx] = 1.0

        return mask

    def _sample_block_params(self, B, device, mode: str, level_type: str):
        """每个块设置局部采样参数
        params: [tx, ty, scale, theta]
        同时采用粗细双层次采样：多层参数图"""
        raw_params = torch.rand(B, 4, 1, 1, device=device) * 2 - 1

        if level_type == 'coarse':
            raw_params[:, 2:4] *= 0.5
        elif level_type == 'fine':
            raw_params[:, 0:2] *= 0.5
        else:
            raise ValueError(f'Unknown level_type: {level_type}')

        if mode == 'random':
            amp_ratio = self.random_aug_ratio
        elif mode == 'guided':
            amp_ratio = self.guided_aug_ratio
        else:
            raise ValueError(f'Unknown transform mode: {mode}')

        return raw_params * amp_ratio
    
    def _build_level_coords(self, H, W):
        """
        在给定层级下生成不规则块划分坐标
        grid_res 表示这一层希望有多少段，而不是规则网格大小
        """
        # 在高度方向随机选 grid_res-1 个切分点
        y_axis = [0] + np.random.choice(
            list(range(1, H)), self.num_block - 1, replace=False
        ).tolist() + [H]

        # 在宽度方向随机选 grid_res-1 个切分点
        x_axis = [0] + np.random.choice(
            list(range(1, W)), self.num_block - 1, replace=False
        ).tolist() + [W]

        y_axis.sort()
        x_axis.sort()

        coords = []
        for i, x1 in enumerate(x_axis[1:]):
            for j, y1 in enumerate(y_axis[1:]):
                coords.append((x_axis[i], x1, y_axis[j], y1))

        return coords, x_axis, y_axis
    
    def build_single_level_control_map(
        self,
        x: torch.Tensor,
        importance_map: torch.Tensor = None,
        transform_mode: str = 'guided',
        level_type: str = 'coarse'
    ):
        B, C, H, W = x.shape
        device = x.device

        coords, x_axis, y_axis = self._build_level_coords(H, W)
        gh = len(x_axis) - 1
        gw = len(y_axis) - 1
        control_map = torch.zeros(B, 4, gh, gw, device=device)

        if importance_map is None or (not self.use_importance_map):
            stable_map = None
            scores = torch.ones(B, len(coords), device=device)
        else:
            # print(f'coarse_importance_blur_kernel: {self.coarse_importance_blur_kernel}')
            # print(f'fine_importance_blur_kernel: {self.fine_importance_blur_kernel}')

            if level_type == 'coarse':
                stable_map = self.coarse_importance_blur(importance_map.detach())
            else:
                stable_map = self.fine_importance_blur(importance_map.detach())

            # stable_map, _ = self.stabilize_importance_map(level_imp)
            scores, stable_map = self._compute_block_scores(stable_map, coords, H, W)

        activation_mask = self._sample_activation_mask(scores, mode=transform_mode)

        idx = 0
        for i in range(gh):
            for j in range(gw):
                params = self._sample_block_params(B, device, mode=transform_mode, level_type=level_type)
                active = activation_mask[:, idx].view(B, 1, 1, 1)

                params = active * params + (1.0 - active) * (self.inactive_scale * params)
                control_map[:, :, i:i+1, j:j+1] = params
                idx += 1

        info = {
            'coords': coords,
            'scores': scores.detach() if scores is not None else None,
            'activation_mask': activation_mask.detach(),
            'grid_shape': (gh, gw),
            'level_type': level_type,
        }
        return control_map, info

    def build_control_param_map(
        self,
        x: torch.Tensor,
        importance_map: torch.Tensor = None,
        transform_mode: str = 'guided'
    ):
        """ 生成粗细双层级控制参数图
        coarse 层参数图用于全局分布扩展
        fine 层参数图用于局部分布扩展
        """
        B, C, H, W = x.shape

        coarse_map, coarse_info = self.build_single_level_control_map(
            x,
            importance_map=importance_map,
            transform_mode=transform_mode,
            level_type='coarse'
        )

        fine_map, fine_info = self.build_single_level_control_map(
            x,
            importance_map=importance_map,
            transform_mode=transform_mode,
            level_type='fine'
        )

        coarse_up = self.upsample_param_map(coarse_map, (H, W))
        fine_up = self.upsample_param_map(fine_map, (H, W))

        coarse_ratio = self.coarse_fuse_ratio
        fine_ratio = 1.0 - coarse_ratio

        fused_param_map = torch.zeros_like(coarse_up)

        # tx, ty 更依赖 coarse 层
        fused_param_map[:, 0:2] = coarse_ratio * coarse_up[:, 0:2] + fine_ratio * fine_up[:, 0:2]

        # scale, theta 更依赖 fine 层
        fused_param_map[:, 2:4] = fine_ratio * coarse_up[:, 2:4] + coarse_ratio * fine_up[:, 2:4]

        info = {
            'coarse_info': coarse_info,
            'fine_info': fine_info,
            'coarse_control_map': coarse_map.detach(),
            'fine_control_map': fine_map.detach(),
            'fused_control_map': fused_param_map.detach(),
        }
        return fused_param_map, info

    def upsample_param_map(self, control_map: torch.Tensor, size_hw):
        H, W = size_hw
        return F.interpolate(control_map, size=(H, W), mode='bicubic', align_corners=False)

    def param_map_to_grid(self, param_map: torch.Tensor):
        B, _, H, W = param_map.shape
        device = param_map.device
        tx = param_map[:, 0] * self.max_disp
        ty = param_map[:, 1] * self.max_disp
        scale = 1.0 + param_map[:, 2] * self.max_scale
        scale = scale.clamp(min=0.7)
        theta = param_map[:, 3] * self.max_theta

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.expand(B, -1, -1)
        grid_y = grid_y.expand(B, -1, -1)

        rot_x = grid_x * torch.cos(theta) - grid_y * torch.sin(theta)
        rot_y = grid_x * torch.sin(theta) + grid_y * torch.cos(theta)

        scaled_x = rot_x * (1.0 / scale)
        scaled_y = rot_y * (1.0 / scale)

        final_grid_x = scaled_x + tx
        final_grid_y = scaled_y + ty
        final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)
        return final_grid

    def warp_from_param_map(self, x: torch.Tensor, param_map: torch.Tensor):
        final_grid = self.param_map_to_grid(param_map)
        x_transformed = F.grid_sample(
            x, final_grid, mode='bilinear', padding_mode='reflection', align_corners=False
        )
        return x_transformed, final_grid

    def blocktransform(self, x: torch.Tensor, importance_map: torch.Tensor = None,
                       return_info: bool = False, transform_mode: str = 'guided'):
        B, C, H, W = x.shape

        # module1
        param_map, info = self.build_control_param_map(
            x, importance_map=importance_map, transform_mode=transform_mode
        )
        x_transformed, final_grid = self.warp_from_param_map(x, param_map)

        # if return_info:
        #     info.update({
        #         'control_map': control_map.detach(),
        #         'param_map_raw': param_map_raw.detach(),
        #         'param_map_used': param_map.detach(),
        #         'final_grid': final_grid.detach(),
        #         'x_transformed': x_transformed.detach(),
        #     })
        #     return x_transformed, info
        return x_transformed

    def transform(self, x, importance_map=None, **kwargs):

        guided_copies = int(round(self.num_copies * self.guided_ratio)) if self.use_importance_map else 0
        guided_copies = min(max(guided_copies, 0), self.num_copies)
        random_copies = self.num_copies - guided_copies

        outputs = []
        # print(f'random_copies: {random_copies}')
        for _ in range(random_copies):
            outputs.append(self.blocktransform(x, importance_map=None, transform_mode='random', **kwargs))
        for _ in range(guided_copies):
            outputs.append(self.blocktransform(x, importance_map=importance_map, transform_mode='guided', **kwargs))

        return torch.cat(outputs, dim=0)
    
    # def transform(self, x, importance_map=None, **kwargs):
    #     """
    #     Scale the input for BlockShuffle
    #     """
    #     return torch.cat([
    #         self.blocktransform(x, importance_map=importance_map, **kwargs)
    #         for _ in range(self.num_copies)
    #     ])

    # 保存块可视化
    def _tensor_to_pil(self, img_tensor, mean=None, std=None):

        """
        img_tensor: [C, H, W], torch tensor
        mean/std: 若输入做过标准化，可传入用于反归一化
        """
        img = img_tensor.detach().cpu().clone()

        if mean is not None and std is not None:
            mean = torch.tensor(mean).view(-1, 1, 1)
            std = torch.tensor(std).view(-1, 1, 1)
            img = img * std + mean

        img = img.clamp(0, 1)
        return to_pil_image(img)
    
    def _draw_block_on_full_image(self, pil_img, x0, y0, x1, y1, color="red", width=3):
        img_draw = pil_img.copy()
        draw = ImageDraw.Draw(img_draw)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
        return img_draw
    
    def _concat_images_horizontally(self, images, padding=10, bg_color=(255, 255, 255)):
        """
        images: PIL Image list
        """
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths) + padding * (len(images) - 1)
        max_height = max(heights)

        canvas = Image.new("RGB", (total_width, max_height), color=bg_color)

        x_offset = 0
        for img in images:
            canvas.paste(img, (x_offset, 0))
            x_offset += img.size[0] + padding

        return canvas
    
    def save_block_visualizations(
        self,
        x,
        save_dir,
        mean=None,
        std=None,
        sample_idx=0,
        save_full_images=True,
        prefix="block_vis"
    ):
        """
        对输入 x 执行一次整图块划分+整图形变，
        然后将每个块保存为：
        [原始大图(高亮当前块) | 原始块 | 形变后的块]

        参数:
            x: [B, C, H, W]
            save_dir: 保存目录
            mean, std: 如果输入做过 normalize，则传入反归一化参数
            sample_idx: 取 batch 中第几张图保存
            save_full_images: 是否额外保存整图原图/整图形变图
            prefix: 文件名前缀
        """
        os.makedirs(save_dir, exist_ok=True)

        assert x.dim() == 4, "x 必须是 [B, C, H, W]"
        assert 0 <= sample_idx < x.shape[0], "sample_idx 超出 batch 范围"

        with torch.no_grad():
            x_transformed, info = self.blocktransform(x, return_info=True)

        x_axis = info["x_axis"]
        y_axis = info["y_axis"]

        # 只取 batch 中一张图做可视化
        orig_img_tensor = x[sample_idx]
        trans_img_tensor = x_transformed[sample_idx]

        orig_full_pil = self._tensor_to_pil(orig_img_tensor, mean=mean, std=std)
        trans_full_pil = self._tensor_to_pil(trans_img_tensor, mean=mean, std=std)

        if save_full_images:
            orig_full_pil.save(os.path.join(save_dir, f"{prefix}_full_original.png"))
            trans_full_pil.save(os.path.join(save_dir, f"{prefix}_full_transformed.png"))

        block_count = 0
        block_meta = []

        # 按行优先顺序保存
        for row, y1 in enumerate(y_axis[1:]):
            y0 = y_axis[row]
            for col, x1 in enumerate(x_axis[1:]):
                x0 = x_axis[col]

                # 裁原始块 / 形变后块
                orig_patch_tensor = orig_img_tensor[:, y0:y1, x0:x1]
                trans_patch_tensor = trans_img_tensor[:, y0:y1, x0:x1]

                orig_patch_pil = self._tensor_to_pil(orig_patch_tensor, mean=mean, std=std)
                trans_patch_pil = self._tensor_to_pil(trans_patch_tensor, mean=mean, std=std)

                # 大图高亮当前块
                full_with_box_pil = self._draw_block_on_full_image(
                    orig_full_pil, x0, y0, x1, y1, color="red", width=3
                )

                # 拼接图：[原始大图带框 | 原始块 | 形变后块]
                compare_pil = self._concat_images_horizontally(
                    [full_with_box_pil, orig_patch_pil, trans_patch_pil],
                    padding=12
                )

                save_name = f"{prefix}_r{row}_c{col}.png"
                save_path = os.path.join(save_dir, save_name)
                compare_pil.save(save_path)

                block_meta.append({
                    "row": row,
                    "col": col,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "save_path": save_path
                })

                block_count += 1

        print(f"[SmoothStructuredSIA] 已保存 {block_count} 个块的可视化结果到: {save_dir}")
        return {
            "save_dir": save_dir,
            "num_blocks": block_count,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "block_meta": block_meta
        }