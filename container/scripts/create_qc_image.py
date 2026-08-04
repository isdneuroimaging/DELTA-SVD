#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re, os
import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

colorExclusions = 'blue'
cmap = ListedColormap(["red", colorExclusions, "magenta"])


def crop_nonzero(images, mask, border, thr=0, verbose = True):
    
    nz =np.nonzero(mask > thr)
    if not all([len(x) for x in nz]):
        print('Warning: Mask is empty!')
        print('Returning size of image')
        idxLow  = np.asarray([0,0,0])
        idxHigh = np.asarray(mask.shape)
    else:
        idxLow  = np.asarray([min(nz[0]),min(nz[1]),min(nz[2])])
        idxHigh = np.asarray([max(nz[0]),max(nz[1]),max(nz[2])])
    
    if verbose: print("index low: ", idxLow)
    if verbose: print("index high:", idxHigh)

    if border is not None:
        if verbose: print(f"Adding border of {border} voxels")
        idxLow  = np.maximum(idxLow-border, [0,0,0])
        idxHigh = np.minimum(idxHigh+border, np.array(mask.shape[0:3])-1)
        if verbose: print("index low: ", idxLow)
        if verbose: print("index high:", idxHigh)
    
    ranges = idxHigh - idxLow + 1
    
    if ranges[2]<6:
        idxLow[2]  = np.maximum(idxLow[2]-np.floor((6-ranges[2])/2), 0)
        idxHigh[2] = np.minimum(idxHigh[2]+np.ceil((6-ranges[2])/2), np.array(mask.shape[2])-1)

    for iImg, img in enumerate(images):
        images[iImg] = img[idxLow[0]:idxHigh[0]+1,
                        idxLow[1]:idxHigh[1]+1,
                        idxLow[2]:idxHigh[2]+1]
    
    return images, idxLow, idxHigh


def create_qc_image(fnamesBG, vlim, labels=None, fnameMask=None, fnameBmask=None, fnameCropMask=None, animate=True, depth=1, addLegends=(0,0)):

    
    nBG = len(fnamesBG)
    if fnameMask is None:
        raise ValueError('Currently the use without an overlay mask is not implemented! Please provide argument "fnameMask"!')
    if fnameBmask is None:
        fnames = fnamesBG + [fnameMask]
    else:
        if not isinstance(fnameBmask, list):
            fnameBmask = [fnameBmask]
        fnames = fnamesBG + [fnameMask] + fnameBmask
    images = []
    for fn in fnames:
        nii = nib.load(fn)
        images.append( nii.get_fdata() )
    if fnameBmask is None:
        images.append( (images[0]>0).astype(np.uint8) )

    border=1
    if fnameCropMask:
        niiCropMask = nib.load(fnameCropMask)
        imgCropMask = niiCropMask.get_fdata()
    else:
        imgCropMask = images[-1]
    images, _, _ = crop_nonzero(images, imgCropMask, border=border, thr=0, verbose = False)

    # equally spaced slices, half a gap from the image border
    nSlices = 8
    zSize = images[0].shape[2]
    idxSlices = np.ceil(np.round(np.linspace(0,zSize-1, nSlices+2))).astype(np.int32)
    idxSlices = idxSlices[1:-1]

    # slice, orient, and reshape into one row of panels
    # 'nii' here is the loop variable leaked from above (the last file in 'fnames',
    # a mask): safe only because all inputs share the same orientation.
    axcodes = nib.aff2axcodes(nii.affine)
    for iImg, img in enumerate(images):    
        img = img[0:,0:,idxSlices]
        if axcodes == ('R','A','S'):
            img = np.fliplr(np.rot90(img))
        elif axcodes == ('L','A','S'):
            img = np.rot90(img)
        else:
            print(f'Warning during creation of QC-image: Axis direction codes "{axcodes}" might not be handled correctly!')
        img = img.reshape(img.shape[0],img.shape[1]*nSlices, order='F')
        images[iImg] = ndimage.zoom(img, 5, order=0)

    imagesBG = images[0:nBG]
    imgMask = np.ma.masked_array(images[nBG].astype(np.uint8), images[nBG]==0) 
    imgBmask = [img>=1 for img in images[nBG+1:]] #--- transforms interpolate the masks, so threshold here at the latest


    for iBG, imgBG in enumerate(imagesBG):
        vlimVol = vlim[iBG]
        #-- figure sized to exactly the image pixels
        height, width = imgBG.shape
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.imshow(imgBG, cmap=plt.cm.gray, aspect='equal', interpolation='nearest', vmin=vlimVol[0], vmax=vlimVol[1])
        ax.imshow(imgMask, cmap=cmap, aspect='equal', interpolation='nearest', vmin=0.5, vmax=3.5, alpha = 0.4) #--- plt.cm.bwr
        if iBG<len(imgBmask):
            imgBmaskT = imgBmask[iBG]
        else:
            imgBmaskT = imgBmask[-1]
        ax.contour(imgBmaskT, levels=[0.5], colors='cyan', linestyles='solid', linewidths=1, alpha=1, antialiased = True)
        fontSz = np.ceil(imgBmaskT.shape[0]/25)
        dist = np.ceil(imgBmaskT.shape[0]/40)
        if labels is not None:
            label = labels[iBG]
        else:
            label = fnamesBG[iBG].split(os.path.sep)
            label = os.path.sep.join(label[-depth:])
        ax.text(dist, fontSz+dist, label, fontsize=fontSz, color='white')
        if iBG<addLegends[0]:
            legend_elements = [patches.Rectangle((0, 0), 0.1, 0.1, facecolor='red', edgecolor='k', alpha=0.7, label='white matter skeleton'),
                               patches.Rectangle((0, 0), 0.1, 0.1, facecolor=colorExclusions, edgecolor='k', alpha=0.7, label='excluded areas'),
                               plt.Line2D([0], [0], linewidth=2, color='cyan', label='brain mask (contour)', markersize=10)]
            if addLegends[1]==0:
                legend_elements.pop(1)
            ax.legend(handles=legend_elements, loc='upper right',fontsize=fontSz, facecolor='black', framealpha=0.5, labelcolor='white')
        # mutates the caller's fnamesBG list in place and writes the PNG beside the NIfTI input
        fnamesBG[iBG] = re.sub(r'\.nii(\.gz)?$', '.png', fnamesBG[iBG])
        plt.savefig(fnamesBG[iBG], transparent=False, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    if animate:
        frames = [iio.imread(fn) for fn in fnamesBG]
        fnPNG = re.sub(r'\.nii(\.gz)?$', '_QC.png', fnameMask)
        iio.imwrite(fnPNG, frames, format="PNG", duration=1000)

        return fnPNG
    
    else:
        return fnamesBG
