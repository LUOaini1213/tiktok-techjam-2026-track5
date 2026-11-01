"""Experimental local summary; the frozen V3 extractor is deliberately untouched.

Local DINO patch dispersion is NOT a localization mask. Native corner statistics
are a bounded extra CPU cost; no second backbone, fine-tuning or resizing is used
for forensic crops. The first 924 dimensions reproduce the V3 fused descriptor.
"""
from __future__ import annotations
import numpy as np
import torch
from .augment import to_rgb, jpeg_compress, down_up_resize
from .features import load_dino, clip_embed_batch_dual, forensic_vector
from .forensic_v4_fast import forensic_vector_fast

GLOBAL_DIM = 924
LOCAL_DIM = 824  # 384 patch std + 384 patch max absolute deviation + 2*28 corner stats
V4_DIM = GLOBAL_DIM + LOCAL_DIM


def pool_patch_tokens(tokens):
    if tokens.ndim != 3 or tokens.shape[1] < 1 or tokens.shape[2] != 384:
        raise ValueError('Expected [batch, patches, 384] DINO tokens')
    t = torch.nn.functional.normalize(tokens.float(), dim=-1)
    mean = t.mean(dim=1, keepdim=True)
    # Population std is finite even for a degenerate single-patch input.
    return torch.cat([t.std(dim=1, correction=0), (t-mean).abs().amax(dim=1)], dim=-1)


def corner_summary(image):
    image = to_rgb(image); w,h=image.size
    cw,ch=min(w,256),min(h,256)
    boxes=[(0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch)]
    f=np.stack([forensic_vector_fast(image.crop((x,y,x+cw,y+ch))) for x,y in boxes])
    return np.r_[f.std(axis=0), f.max(axis=0)].astype(np.float32)


def center_forensic(image):
    # Same pixels/padding as V3, but crop BEFORE float32 conversion so a native
    # multi-megapixel photograph does not allocate a full-resolution float array.
    w,h=image.size; x=max(0,(w-256)//2); y=max(0,(h-256)//2)
    return forensic_vector(image.crop((x,y,min(w,x+256),min(h,y+256))))


def embed_v4_batch(images, model_id='openai/clip-vit-base-patch32', *,
                   use_tta=True, tta_jpeg_quality=70, tta_resize_scale=.5):
    images=[to_rgb(im) for im in images]
    if not images: return np.empty((0,V4_DIM),dtype=np.float32)
    clip_views=[]
    n_views=3 if use_tta else 1
    for image in images:
        clip_views.append(image)
        if use_tta: clip_views.extend([jpeg_compress(image,tta_jpeg_quality),down_up_resize(image,tta_resize_scale)])
    clip=clip_embed_batch_dual(clip_views,model_id)['proj'].reshape(len(images),n_views,512).mean(axis=1)
    clip=clip/(np.linalg.norm(clip,axis=1,keepdims=True)+1e-8)
    processor,model,device=load_dino()
    pixel=processor(images=images,return_tensors='pt')['pixel_values'].to(device)
    with torch.inference_mode():
        out=model(pixel_values=pixel)
        cls=torch.nn.functional.normalize(out.pooler_output.float(),dim=-1).cpu().numpy()
        patches=pool_patch_tokens(out.last_hidden_state[:,1:,:]).cpu().numpy()
    # Match the second mean-vector normalization in V3's embed_one_dual.
    cls=cls/(np.linalg.norm(cls,axis=1,keepdims=True)+1e-8)
    center=np.stack([center_forensic(im) for im in images])
    corner=np.stack([corner_summary(im) for im in images])
    X=np.concatenate([clip,cls,center,patches,corner],axis=1).astype(np.float32)
    if X.shape!=(len(images),V4_DIM) or not np.isfinite(X).all(): raise ValueError('Invalid V4 features')
    return X


def embed_v4(image, **kwargs):
    return embed_v4_batch([image], **kwargs)[0]
