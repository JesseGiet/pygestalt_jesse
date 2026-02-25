"""Patches.
"""
import os
import sys
import numpy as np
from numpy.typing import ArrayLike


def segment(X:ArrayLike, h:ArrayLike, l:float, w:float) -> ArrayLike:
    """Line segment function.

    Args:
        X: coordinates of evaluation, of dimension `?x2`.
        h: tangent direction of the segment, which is perpendicular to the normal direction.
        l: length of the segment.
        w: width of the segment.

    Returns:
        function values at the given coordinates.
    """
    h = h / np.linalg.norm(h)
    g = np.asarray([-h[1], h[0]])
    return (np.abs(X @ g) < w/2) * (np.abs(X @ h) < l/2)

# def gabor(x, *,f:float, σ2:float, θ:float):
#     """Gabor function.
#     """
#     return (1+np.exp(-(x[0]**2+x[1]**2)/(2*σ2))*np.cos(2*np.pi*f*(x[1]*np.cos(θ)-x[0]*np.sin(θ))))/2

# vgabor = np.vectorize(gabor)


def generate_image(Ps:ArrayLike, Hs:ArrayLike=None, *, N:int, pfunc:callable) -> ArrayLike:
    """Generate a pixel image of random oriented patches, with optional tangent directions.

    Args:
        Ps: position of balls in the square `[0,1]x[0,1]`, of shape `?x2`
        Hs: tangent direction at `Ps`.
        N: image resolution in pixels.
        pfunc: patch function. `pfunc(z,g)` is the patch function with orientation `g` evaluated at `z` (of shape `?x2`).

    Returns:
        a pixel image.
    """
    I = np.zeros((N,N), dtype=float)

    # meshgrid on [0,1]x[0,1]
    XYg = np.stack(np.meshgrid(range(N), range(N))).reshape(2,-1).T / N
    # relative coordinates of meshgrid points to balls
    Z = XYg[:,:,None] - Ps.T[None,:,:]
    Hlist = []
    # iteration over balls is more efficient than over pixels
    for n in range(Z.shape[-1]):
        z = Z[:,:,n]
        if Hs is not None:
            h = Hs[n] #; h /= np.linalg.norm(h)
        else:
            h = np.random.randn(2) #; h /= np.linalg.norm(h)
        Hlist.append(h) 
        H = np.vstack(Hlist)
        I += pfunc(z,h).reshape(N, N)

    return I, H
