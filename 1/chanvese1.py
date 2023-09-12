# ------------------------------------------------------------------------
# Region Based Active Contour Segmentation
#
# seg = region_seg(I, init_mask, max_its, alpha, display)
#
# Inputs: I           2D image
#         init_mask   Initialization (1 = foreground, 0 = bg)
#         max_its     Number of iterations to run segmentation for
#         alpha       (optional) Weight of smoothing term
#                       higer = smoother.  default = 0.2
#         display     (optional) displays intermediate outputs
#                       default = true
#
# Outputs: seg        Final segmentation mask (1=fg, 0=bg)
#
# Description: This code implements the paper: "Active Contours Without
# Edges" By Chan Vese. This is a nice way to segment images whose
# foregrounds and backgrounds are statistically different and homogeneous.
#
# Example:
# img = imread('tire.tif');
# m = zeros(size(img));
# m(33:33+117, 44:44+128) = 1;
# seg = region_seg(img, m, 500);
#
# Coded by: Shawn Lankton (www.shawnlankton.com)
# ------------------------------------------------------------------------
# 
# Gif montage (save each step in a gif folder):
# convert -delay 50 gif/levelset*.png levelset.gif

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy.signal import convolve2d

h =[

   [ 0.0030 ,   0.0133  ,  0.0219  ,  0.0133 ,   0.0030],
   [ 0.0133 ,   0.0596 ,   0.0983 ,   0.0596  ,  0.0133],
   [ 0.0219  ,  0.0983 ,   0.1621   , 0.0983    ,0.0219],
   [ 0.0133  ,  0.0596   , 0.0983  ,  0.0596  ,  0.0133],
   [ 0.0030   , 0.0133 ,   0.0219  ,  0.0133   , 0.0030]
]
#img = data.astronaut()
#img = rgb2gray(img)

#s = np.linspace(0, 2*np.pi, 400)
#r = 100 + 200*np.sin(s)
#c = 220 + 200*np.cos(s)
#init = np.array([r, c]).T

eps = np.finfo(float).eps


def chanvese(ground_truth,I, init_mask,alpha,betha,gamma, max_its=150,
             thresh=0, color='r', display=False):
    I = I.astype(np.float)
    img=I
    ground_truth= np.squeeze(ground_truth)
    initial_contour=init_mask
    gt=ground_truth

    # Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask)
    phi0=phi
    #plt.imshow(phi)
    #plt.show()
    #print("phi0")
    #print(phi[90:109,90:100])

    if display:
        plt.ion()
        fig, axes = plt.subplots(ncols=2)
        show_curve_and_phi(initial_contour,ground_truth,fig, I, phi, color)
        plt.savefig('levelset_start.png', bbox_inches='tight')

    # Main loop
    its = 0
    stop = False
    prev_mask = init_mask
    c = 0

    while (its < max_its and not stop):
        # Get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= 1.2, phi >= -1.2))
        features_tmp= convolve2d(img, h, 'same')
        featuresx, featuresy = np.gradient(features_tmp)
        features = np.sqrt(np.power(featuresx,2)+np.power(featuresy,2)+eps);
        features = 1 / ( 1 + np.power(features,2) );

        if len(idx) > 0:
            # Intermediate output
            #if display:
            #    if np.mod(its, 50) == 0:
            #        print('iteration: {0}'.format(its))
            #        show_curve_and_phi(initial_contour,ground_truth,fig, I, phi, color)
            """else:
                if np.mod(its, 10) == 0:
                    print('iteration: {0}'.format(its))"""

            # Find interior and exterior mean
            upts = np.flatnonzero(phi <= 0)  # interior points
            vpts = np.flatnonzero(phi > 0)  # exterior points
            u = np.sum(I.flat[upts]) / (len(upts) + eps)  # interior mean
            v = np.sum(I.flat[vpts]) / (len(vpts) + eps)  # exterior mean
            dimy, dimx = phi.shape;  
            #print(idx)
            curvature,FdotGrad,normGrad = get_curvature(phi, idx, features)

            """y,x = np.unravel_index(idx, (dimy,dimx))
            #idx2 = np.ravel_multi_index(np.array([np.transpose(y),np.transpose(x)]),phi.shape)   

            #print(y)
            #print(x)
            ym1 = y-1
            xm1 = x-1
            yp1 = y+1
            xp1 = x+1;
            #ym1(ym1<1) = 1; xm1(xm1<1) = 1;              
            #yp1(yp1>dimy)=dimy; xp1(xp1>dimx) = dimx;   

            idup = np.ravel_multi_index(np.array([np.transpose(yp1),np.transpose(x)]),phi.shape)   
            iddn = np.ravel_multi_index(np.array([np.transpose(ym1),np.transpose(x)]),phi.shape)
            idlt = np.ravel_multi_index(np.array([np.transpose(y),np.transpose(xm1)]),phi.shape)
            idrt = np.ravel_multi_index(np.array([np.transpose(y),np.transpose(xp1)]),phi.shape)
            idul = np.ravel_multi_index(np.array([np.transpose(yp1),np.transpose(xm1)]),phi.shape)
            idur = np.ravel_multi_index(np.array([np.transpose(yp1),np.transpose(xp1)]),phi.shape)
            iddl = np.ravel_multi_index(np.array([np.transpose(ym1),np.transpose(xm1)]),phi.shape)
            iddr = np.ravel_multi_index(np.array([np.transpose(ym1),np.transpose(xp1)]),phi.shape)
            #get gradients
            phi_x  = (-phi.flat[idlt]+phi.flat[idrt])/2
            phi_y  = (-phi.flat[iddn]+phi.flat[idup])/2
            phi_xx = phi.flat[idlt]-2*phi.flat[idx]+phi.flat[idrt];
            phi_yy = phi.flat[iddn]-2*phi.flat[idx]+phi.flat[idup];
            phi_xy = 0.25*phi.flat[iddl]+0.25*phi.flat[idur]-0.25*phi.flat[iddr]-0.25*phi.flat[idul];
            phi_x2 = np.power(phi_x,2);
            phi_y2 = np.power(phi_y,2);
            #compute norm of gradient
            curvature = ((phi_x2*phi_yy + phi_y2*phi_xx - 2*phi_x*phi_y*phi_xy)/ \
            np.power((phi_x2 + phi_y2 +eps),(3/2)));"""
  
            #print(curvature)
            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
            """phi_xm = phi.flat[idx]-phi.flat[idlt];
            phi_xp = phi.flat[idrt]-phi.flat[idx];
            phi_ym = phi.flat[idx]-phi.flat[iddn];
            phi_yp = phi.flat[idup]-phi.flat[idx]; 
            #compute scalar product between the feature image and the gradient of phi
            F_x = 0.5*features.flat[idrt]-0.5*features.flat[idlt]
            F_y = 0.5*features.flat[idup]-0.5*features.flat[iddn]              
            FdotGrad = (np.maximum(F_x,0))*(phi_xp)+ (np.minimum(F_x,0))*(phi_xm) + \
            (np.maximum(F_y,0))*(phi_yp) + (np.minimum(F_y,0))*(phi_ym) """      
            # Force from image information
            F=0
            F1=0
            F2=0
            F3=0
            F4=0
            F5=0
            F1 = (I.flat[idx] - u)**2 - (I.flat[idx] - v)**2
            #print(F1)
            F2= -1*(phi.flat[idx]-phi0.flat[idx])
            #print('*************************************************************')
            #print(F2)
            F3=FdotGrad
            # Force from curvature penalty
            #print(curvature2)
            """normGrad = np.sqrt(np.power( (np.maximum(phi_xm,0)),2)\
            + np.power((np.minimum(phi_xp,0)),2) + np.power((np.maximum(phi_ym,0)),2)\
            + np.power((np.minimum(phi_yp,0)),2 ))
            print(normGrad)
            #print("QQQQQQQQQQQQQQQQQqq")
            #print(normGrad)"""
            #C:\Users\r_beh\OneDrive\Desktop\cardiac-segmentation-master\cardiac-segmentation-master
            # Gradient descent to minimize energy
            F4=  np.multiply(features.flat[idx],normGrad)*curvature
            F5=  np.multiply(features.flat[idx],normGrad)
            #print(F5)
            #dphidt = alpha*F1 / np.max(np.abs(F1)) + betha * curvature+ gamma*F2/(np.max(np.abs(F2))+eps)
            #+0.2*F3/(np.max(np.abs(F3))+eps)
            #+0.1* F2 / (np.max(np.abs(F2))+eps)
            #+ F3/(np.max(np.abs(F3))+eps)
            dphidt = alpha*F3/(np.max(np.abs(F3))+eps) + betha*F4/(np.max(np.abs(F4))+eps) -gamma*  F5/(np.max(np.abs(F5))+eps)+0*F2/(np.max(np.abs(F2))+eps)
            #+ 0.001*F1/(np.max(np.abs(F1))+eps)
            #dphidt=  F3/(np.max(np.abs(F3))+eps) + F4/(np.max(np.abs(F4))+eps) - F5/(np.max(np.abs(F5))+eps)
            # Maintain the CFL condition
            dt = 0.45 / (np.max(np.abs(dphidt)) + eps)
            
            # Evolve the curve
            phi.flat[idx] += dt * dphidt

            # Keep SDF smooth
            phi = sussman(phi, 0.5)
            

            new_mask = phi # <= 0
            c = convergence(prev_mask, new_mask, thresh, c)

            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

        else:
            break

    # Final output
    if display:
        show_curve_and_phi(initial_contour,ground_truth,fig, I, phi, color)
        plt.savefig('levelset_end.png', bbox_inches='tight')

    # Make mask from SDF
    seg = phi <= 0  # Get mask from levelset
    return seg, phi, its


# ---------------------------------------------------------------------
# ---------------------- AUXILIARY FUNCTIONS --------------------------
# ---------------------------------------------------------------------

def bwdist(a):
    """
    Intermediary function. 'a' has only True/False vals,
    so we convert them into 0/1 values - in reverse.
    True is 0, False is 1, distance_transform_edt wants it that way.
    """
    return nd.distance_transform_edt(a == 0)


# Displays the image with curve superimposed
def show_curve_and_phi(initial,groundt,fig, I, phi, color):
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray')
    #fig.axes[0].imshow(groundt)
    fig.axes[0].contour(phi, 0, colors=color)
    fig.axes[0].contour(initial, 0, colors='b')    
    fig.axes[0].plot(groundt[:,0],groundt[:,1],'g')
    fig.axes[0].set_axis_off()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.draw()
    
    fig.axes[1].cla()
    fig.axes[0].imshow(I, cmap='gray')
    fig.axes[1].set_axis_off()
    plt.draw()
    
    plt.pause(0.001)


"""def show_curve_and_phi(fig, I, phi, color):
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray')
    fig.axes[0].contour(phi, 0, colors=color)
    fig.axes[0].set_axis_off()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.draw()
    
    fig.axes[1].cla()
    fig.axes[1].imshow(phi)
    fig.axes[1].set_axis_off()
    plt.draw()
    
    plt.pause(0.001)
"""
def im2double(a):
    a = a.astype(np.float)
    a /= np.abs(a).max()
    return a


# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + im2double(init_a) - 0.5
    return phi


# Compute curvature along SDF
def get_curvature(phi, idx, features):
    dimy, dimx = phi.shape
    yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # subscripts
    y = yx[:, 0]
    x = yx[:, 1]

    # Get subscripts of neighbors
    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    # Bounds checking
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # Get indexes for 8 neighbors
    idup = np.ravel_multi_index((yp1, x), phi.shape)
    iddn = np.ravel_multi_index((ym1, x), phi.shape)
    idlt = np.ravel_multi_index((y, xm1), phi.shape)
    idrt = np.ravel_multi_index((y, xp1), phi.shape)
    idul = np.ravel_multi_index((yp1, xm1), phi.shape)
    idur = np.ravel_multi_index((yp1, xp1), phi.shape)
    iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
    iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

    # Get central derivatives of SDF at x,y
    phi_x = -phi.flat[idlt] + phi.flat[idrt]
    phi_y = -phi.flat[iddn] + phi.flat[idup]
    phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
    phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
    phi_xy = 0.25 * (- phi.flat[iddl] - phi.flat[idur] +
                     phi.flat[iddr] + phi.flat[idul])
    phi_x2 = phi_x**2
    phi_y2 = phi_y**2

    # Compute curvature (Kappa)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + eps) ** 1.5) * (phi_x2 + phi_y2) ** 0.5
                 
    phi_xm = phi.flat[idx]-phi.flat[idlt];
    phi_xp = phi.flat[idrt]-phi.flat[idx];
    phi_ym = phi.flat[idx]-phi.flat[iddn];
    phi_yp = phi.flat[idup]-phi.flat[idx]; 
    #compute scalar product between the feature image and the gradient of phi
    F_x = 0.5*features.flat[idrt]-0.5*features.flat[idlt]
    F_y = 0.5*features.flat[idup]-0.5*features.flat[iddn]              
    FdotGrad = (np.maximum(F_x,0))*(phi_xp)+ (np.minimum(F_x,0))*(phi_xm) + \
    (np.maximum(F_y,0))*(phi_yp) + (np.minimum(F_y,0))*(phi_ym)  

    normGrad = np.sqrt(np.power( (np.maximum(phi_xm,0)),2)\
    + np.power((np.minimum(phi_xp,0)),2) + np.power((np.maximum(phi_ym,0)),2)\
    + np.power((np.minimum(phi_yp,0)),2 ))    

    return curvature,FdotGrad,normGrad


# Level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - np.roll(D, 1, axis=1)
    b = np.roll(D, -1, axis=1) - D
    c = D - np.roll(D, -1, axis=0)
    d = np.roll(D, 1, axis=0) - D

    a_p = np.clip(a, 0, np.inf)
    a_n = np.clip(a, -np.inf, 0)
    b_p = np.clip(b, 0, np.inf)
    b_n = np.clip(b, -np.inf, 0)
    c_p = np.clip(c, 0, np.inf)
    c_n = np.clip(c, -np.inf, 0)
    d_p = np.clip(d, 0, np.inf)
    d_n = np.clip(d, -np.inf, 0)

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros_like(D)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_p.flat[D_pos_ind]**2], [b_n.flat[D_pos_ind]**2])), axis=0) +
        np.max(np.concatenate(
            ([c_p.flat[D_pos_ind]**2], [d_n.flat[D_pos_ind]**2])), axis=0)) - 1
    dD.flat[D_neg_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_n.flat[D_neg_ind]**2], [b_p.flat[D_neg_ind]**2])), axis=0) +
        np.max(np.concatenate(
            ([c_n.flat[D_neg_ind]**2], [d_p.flat[D_neg_ind]**2])), axis=0)) - 1

    D = D - dt * sussman_sign(D) * dD
    return D


def sussman_sign(D):
    return D / np.sqrt(D**2 + 1)


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    diff = p_mask - n_mask
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
    return c


if __name__ == "__main__":
    # img = Image.open('img/fauteuil.png')
    #img = Image.open('C:/Users/r_beh/OneDrive/Desktop/brain.png')
    # size = img.size
    # img.resize((size[0]//4, size[1]//4))
    #img = np.array(ImageOps.grayscale(img))
    mask = np.zeros(img.shape)
    mask[60:100,200:300] = 1

    chanvese(img, mask, max_its=150, display=False, alpha=1.0)
