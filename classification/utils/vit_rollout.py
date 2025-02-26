import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2


def rollout(attentions, discard_ratio, head_fusion, device):
    result = torch.eye(attentions[0].size(-1)).to(device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            # indices = indices[indices != 0]
            # flat[0, indices] = 0

            indices = indices.to(torch.int64)
            flat.scatter_(1, indices, 0)

            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0*I)/2

            a = a / a.sum(dim=-1)[...,None]

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[:, 0 ,1  :]
    # metric = sum(1 for i in mask if i > 0.25)
    # In case of 224x224 image, this brings us from 196 to 14
    mask = mask / mask.max(-1)[0][...,None]
    mask = torch.nn.functional.softmax(mask, dim=-1)
    return mask


def last_attn(attentions, head_fusion):
    if head_fusion == "mean":
        attention_heads_fused = attentions[-1].mean(axis=1)
    elif head_fusion == "max":
        attention_heads_fused = attentions[-1].max(axis=1)[0]
    elif head_fusion == "min":
        attention_heads_fused = attentions[-1].min(axis=1)[0]
    else:
        raise "Attention head fusion type Not supported"
    return attention_heads_fused[:,0,1:]


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9, device = "cuda:0"):
        self.model = model.to(device)
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.device = device

        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []
        self.scale = (model.embed_dim // model.blocks[0].attn.num_heads) ** -0.5

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def __call__(self, input_tensor, ret_out):
        self.attentions = []
        with torch.no_grad():
            output = self.model.forward_features(input_tensor)
            # print(torch.topk(torch.nn.functional.softmax(output,dim=-1), 5))
        if "roll" in ret_out:
            return rollout(self.attentions, self.discard_ratio, self.head_fusion, self.device)
        elif "attn" in ret_out:
            return last_attn(self.attentions, self.head_fusion)
        elif "out" in ret_out:
            return output[:,0,:]

        # return  torch.stack(self.attentions).mean(2).numpy(), 0
        