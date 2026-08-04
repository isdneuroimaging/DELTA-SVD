#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
import re, os
import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import imageio.v3 as iio
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# colorExclusions = 'orange'
colorExclusions = 'blue'
cmap = ListedColormap(["red", colorExclusions, "magenta"])
#
def crop_nonzero(images, mask, border, thr=0, onlyZ=False, verbose = True):
    
    #- find ranges containing mask
    nz =np.nonzero(mask > thr)
    if not all([len(x) for x in nz]):
        print('Warning: Mask is empty!')
        print('Returning size of image')
        idxLow  = np.asarray([0,0,0])
        idxHigh = np.asarray(mask.shape)
    else:
        idxLow  = np.asarray([min(nz[0]),min(nz[1]),min(nz[2])])
        idxHigh = np.asarray([max(nz[0]),max(nz[1]),max(nz[2])])
        # ranges = idxHigh - idxLow +1
    
    if verbose: print("index low: ", idxLow)
    if verbose: print("index high:", idxHigh)

    # add border, if possible
    if border is not None:
        if verbose: print(f"Adding border of {border} voxels")
        idxLow  = np.maximum(idxLow-border, [0,0,0])
        idxHigh = np.minimum(idxHigh+border, np.array(mask.shape[0:3])-1)
        if verbose: print("index low: ", idxLow)
        if verbose: print("index high:", idxHigh)
    
    ranges = idxHigh - idxLow + 1
    # print('Ranges:', ranges)
    
    if ranges[2]<6:
        idxLow[2]  = np.maximum(idxLow[2]-np.floor((6-ranges[2])/2), 0)
        idxHigh[2] = np.minimum(idxHigh[2]+np.ceil((6-ranges[2])/2), np.array(mask.shape[2])-1)

    # crop images
    for iImg, img in enumerate(images):
        if onlyZ:
            images[iImg] = img[:,:, idxLow[2]:idxHigh[2]+1]
        else:
            images[iImg] = img[idxLow[0]:idxHigh[0]+1, 
                            idxLow[1]:idxHigh[1]+1,
                            idxLow[2]:idxHigh[2]+1]
    
    return images, idxLow, idxHigh


def create_qc_image(fnamesBG, vlim, labels=None, fnameMask=None, fnameBmask=None, fnameCropMask=None, fnameOut=None, animate=True, depth=1, addLegends=(0,0)):

    
    # Read images
    nBG = len(fnamesBG)
    if fnameMask is None:
        raise ValueError('Currently the use without an overlay mask is not implemented! Plase provide argument "fnameMask"!')
    if fnameBmask is None:
        fnames = fnamesBG + [fnameMask]
    else:
        if not isinstance(fnameBmask, list):
            fnameBmask = [fnameBmask]
        fnames = fnamesBG + [fnameMask] + fnameBmask
    images = []
    # print('N =', len(fnames))
    for fn in fnames:
        nii = nib.load(fn)
        images.append( nii.get_fdata() )
    if fnameBmask is None:
        images.append( (images[0]>0).astype(np.uint8) )

    # Crop images
    border=1
    if fnameCropMask:
        # print(f'\nUSING FOLLOWING MASK FOR CROPPING:\n   {fnameCropMask}\n')
        niiCropMask = nib.load(fnameCropMask)
        imgCropMask = niiCropMask.get_fdata()
    else:
        imgCropMask = images[-1]
    images, _, _ = crop_nonzero(images, imgCropMask, border=border, thr=0, verbose = False)

    # Determine equally spaced slices, whith distance to the image border being half space
    nSlices = 8
    zSize = images[0].shape[2]
    idxSlices = np.ceil(np.round(np.linspace(0,zSize-1, nSlices+2))).astype(np.int32)
    idxSlices = idxSlices[1:-1]
    # print(idxSlices)

    # Extract slices, flip x/y axis, and reshape to 2D image with panels
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
        # images[iImg] = img
        images[iImg] = ndimage.zoom(img, 5, order=0)

    imagesBG = images[0:nBG]
    imgMask = np.ma.masked_array(images[nBG].astype(np.uint8), images[nBG]==0) 
    # imgMaskContour = images[nBG]
    imgBmask = [img>=1 for img in images[nBG+1:]] #--- brain masks are interpolated during transformations and have to be thresholded here latest. (for cropping, a slightly larger mask is even desireable)


    # Visualize each background image
    # print('N =', len(imagesBG))
    for iBG, imgBG in enumerate(imagesBG):
        vlimVol = vlim[iBG]
        #-- create figure with number of pixels corresponding exactly to image shape
        height, width = imgBG.shape
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        #-- Create an axis covering the entire figure space
        ax = fig.add_axes((0, 0, 1, 1))
        #-- Remove the axes ticks and frames
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        #-- Plot image and contour
        ax.imshow(imgBG, cmap=plt.cm.gray, aspect='equal', interpolation='nearest', vmin=vlimVol[0], vmax=vlimVol[1])
        ax.imshow(imgMask, cmap=cmap, aspect='equal', interpolation='nearest', vmin=0.5, vmax=3.5, alpha = 0.4) #--- plt.cm.bwr
        # ax.contour(imgMaskContour, levels=[0.5], colors='red', linestyles='solid', linewidths=1, alpha=0.4, antialiased = True)
        if iBG<len(imgBmask):
            imgBmaskT = imgBmask[iBG]
        else:
            imgBmaskT = imgBmask[-1]
        ax.contour(imgBmaskT, levels=[0.5], colors='cyan', linestyles='solid', linewidths=1, alpha=1, antialiased = True)
        #-- add label (currently the file name)
        fontSz = np.ceil(imgBmaskT.shape[0]/25)
        dist = np.ceil(imgBmaskT.shape[0]/40)
        if labels is not None:
            label = labels[iBG]
        else:
            label = fnamesBG[iBG].split(os.path.sep)
            label = os.path.sep.join(label[-depth:])
        # print('\n',label)
        # print(vlimVol)
        ax.text(dist, fontSz+dist, label, fontsize=fontSz, color='white')
        #-- add legend
        if iBG<addLegends[0]:
            #-- add legend (manually add markers and text)
            # ax.text(dist, dist+(fontSz+dist)*2, u"\u25A0", fontsize=fontSz, color='red')
            # ax.text(dist, dist+(fontSz+dist)*3, u"\u25B1", fontsize=fontSz, color='cyan')
            # ax.text(dist*4, dist+(fontSz+dist)*2, "white matter skeleton", fontsize=fontSz, color='white')
            # ax.text(dist*4, dist+(fontSz+dist)*3, "mask contour", fontsize=fontSz, color='white')
            #-- add legend (use legend function from Matplotlib)
            legend_elements = [patches.Rectangle((0, 0), 0.1, 0.1, facecolor='red', edgecolor='k', alpha=0.7, label='white matter skeleton'),
                               patches.Rectangle((0, 0), 0.1, 0.1, facecolor=colorExclusions, edgecolor='k', alpha=0.7, label='excluded areas'),
                               plt.Line2D([0], [0], linewidth=2, color='cyan', label='brain mask (contour)', markersize=10)]
            if addLegends[1]==0:
                legend_elements.pop(1)
            ax.legend(handles=legend_elements, loc='upper right',fontsize=fontSz, facecolor='black', framealpha=0.5, labelcolor='white')
        #-- Save
        fnamesBG[iBG] = re.sub(r'\.nii(\.gz)?$', '.png', fnamesBG[iBG])
        plt.savefig(fnamesBG[iBG], transparent=False, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # Create an animated PNG using imageio
    if animate:
        frames = [iio.imread(fn) for fn in fnamesBG]
        if fnameOut is None:
            fnPNG = re.sub(r'\.nii(\.gz)?$', '_QC.png', fnameMask)
        else:
            fnPNG = fnameOut
        iio.imwrite(fnPNG, frames, format="PNG", duration=1000)

        return fnPNG
    
    else:
        return fnamesBG
