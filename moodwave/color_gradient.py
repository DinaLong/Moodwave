
import numpy as np
import matplotlib as mpl

# Given two hex colors, returns a normalized RGB color gradient with n colors.
def get_color_gradient(c1, c2, n):
    c1_rgb = np.array(mpl.colors.to_rgb(c1))/255
    c2_rgb = np.array(mpl.colors.to_rgb(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = np.array([((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts])
    rgb_colors = rgb_colors/np.sum(rgb_colors, axis=1).reshape((n, 1)) + 0.1
    return rgb_colors