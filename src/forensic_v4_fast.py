"""Vectorized corner descriptor, same 28 statistics as the frozen V3 scalar code.

Only experimental V4 corners use this helper. The global 924-D descriptor keeps
V3's original routine. No change to crop scale, filters, DCT bands or FFT rings.
"""
import cv2
import numpy as np
from scipy.fftpack import dct
from .features import _forensic_work,_luma,_safe_corr


def forensic_vector_fast(image):
    rgb=_forensic_work(image)/255.; gray=_luma(rgb); n=gray.shape[0]
    residual=gray-cv2.GaussianBlur(gray,(5,5),1.)
    r=[residual.mean(),residual.std(),np.abs(residual).mean(),np.percentile(residual,5),np.percentile(residual,95)]
    blocks=gray.reshape(n//8,8,n//8,8).transpose(0,2,1,3)
    coeff=dct(dct(blocks,axis=-2,norm='ortho'),axis=-1,norm='ortho')
    low=np.abs(coeff[...,:2,:2]).mean(axis=(-2,-1))
    high=np.abs(coeff[...,2:,2:]).mean(axis=(-2,-1))
    energy=(high/(low+1e-6)).ravel().astype(np.float32)
    d=[energy.mean(),energy.std(),np.percentile(energy,25),np.percentile(energy,75),np.percentile(energy,95)]
    mag=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    yy,xx=np.ogrid[:n,:n]; radii=np.sqrt((yy-n//2)**2+(xx-n//2)**2); rings=[]
    for lo,hi in ((0,.125),(.125,.25),(.25,.5),(.5,.75),(.75,1),(1,1.5)):
        mask=(radii>=lo*n/2)&(radii<hi*n/2)
        rings.append(mag[mask].mean() if mask.any() else 0.)
    color=[*rgb.mean(axis=(0,1)),*rgb.std(axis=(0,1)),
           _safe_corr(rgb[...,0],rgb[...,1]),_safe_corr(rgb[...,1],rgb[...,2]),_safe_corr(rgb[...,0],rgb[...,2])]
    extra=[cv2.Laplacian(gray,cv2.CV_32F).var(),gray.mean(),gray.std()]
    return np.nan_to_num(np.asarray([*r,*d,*rings,*color,*extra],dtype=np.float32),nan=0,posinf=0,neginf=0)
