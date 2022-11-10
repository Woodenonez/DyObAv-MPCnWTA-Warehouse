import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import casadi as cs

print(random.sample(list(range(1,11)), k=5))
sys.exit(0)

def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    out = []
    for i in range(1-len(kernel),len(signal)):
        out.append( np.dot(signal[max(0,i):min(i+len(kernel),len(signal))], kernel[max(-i,0):len(signal)-i*(len(signal)-len(kernel)<i)]) )
    return out

base_weight = [200, 100, 50, 20, 10, 8, 5, 4, 3, 2] + [1]*10
kernel = [1, 0.5, 0.5]
for kt in range(20):
    onehot_signal = [0]*20
    onehot_signal[kt] = base_weight[kt]
    horizon_weight = convolve_1d(onehot_signal, kernel=kernel)
    print(horizon_weight)
sys.exit(0)

a = cs.DM([[1,1,1], [2,3,-4]]).T
print(a)
b = cs.mtimes(a, cs.DM([1,1]))
result = 1
for i in range(b.shape[0]):
    result *= cs.fmax(0, b[i])
print(result)
sys.exit(0)


boundary = [(0,0), (0,10), (12,10), (12,0)]
obstacle_list = [[(1,1), (1,3), (3,3), (3,1)],
                 [(5,5), (7,5), (5,7)]]
width  = max(np.array(boundary)[:,0]) - min(np.array(boundary)[:,0])
height = max(np.array(boundary)[:,1]) - min(np.array(boundary)[:,1])

fig, ax = plt.subplots(figsize=(width, height), dpi=100)
ax.set_aspect('equal')
ax.axis('off')
ax.plot(np.array(boundary)[:,0], np.array(boundary)[:,1], 'r-')
for coords in obstacle_list:
    x, y = np.array(coords)[:,0], np.array(coords)[:,1]
    plt.fill(x, y, color='k')

fig.tight_layout(pad=0)

fig.canvas.draw()
image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

print(fig_size, image_from_plot.shape)
plt.close()

sys.exit(0)

def get_closest_edge_point(original_point:np.ndarray, occupancy_map:np.ndarray) -> np.ndarray:
    if len(original_point.shape) == 1:
        original_point = original_point[np.newaxis,:]
    if len(occupancy_map.shape) != 2:
        raise ValueError(f'Input map should be 2-dim (got {len(occupancy_map.shape)}-dim).')
    if original_point.shape[1] != 2:
        raise ValueError(f'Input point shape incorrect (should be 2, got {original_point.shape[1]}).')
    def np_dist_map(centre, base_matrix): # centre's size is nx2, base_matrix is wxh
        npts = centre.shape[0]
        x = np.arange(0, base_matrix.shape[1])
        y = np.arange(0, base_matrix.shape[0])
        x, y = np.meshgrid(x, y)
        x, y = x[:, :, np.newaxis], y[:, :, np.newaxis]
        xc_full_mtx = np.ones_like(x).repeat(npts, axis=2) * np.expand_dims(centre[:,0], axis=(0,1))
        yc_full_mtx = np.ones_like(y).repeat(npts, axis=2) * np.expand_dims(centre[:,1], axis=(0,1))
        base_matrix = (x.repeat(npts, axis=2)-xc_full_mtx)**2 + (y.repeat(npts, axis=2)-yc_full_mtx)**2
        return base_matrix/base_matrix.max()
    occupancy_map /= np.amax(occupancy_map, axis=(0,1), keepdims=True)
    edge_map = np.array(Image.fromarray(occupancy_map.astype('uint8'), 'L')
                        .filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.FIND_EDGES))
    edge_map[edge_map>0] = 1
    dist_map = np_dist_map(original_point, occupancy_map) * edge_map[:,:,np.newaxis].repeat(original_point.shape[0], axis=2)
    dist_map[dist_map==0] = np.max(dist_map)
    closest_edge_point = []
    for i in range(original_point.shape[0]):
        closest_edge_point.append(np.unravel_index(np.argmin(dist_map[:,:,i], axis=None), dist_map[:,:,i].shape))
    return np.array(closest_edge_point)

bad_point = np.array([[65, 40], [60,35]])
map = np.zeros((100,100))
map[30:61, 50:71] = 1
map[20:41, 20:41] = 1
map[35:45, 70:80] = 1

good_point = get_closest_edge_point(bad_point, map)

print(good_point)

_, [ax1, ax2] = plt.subplots(1,2)
[ax.plot(bad_point[:, 0],  bad_point[:, 1],  'rx') for ax in [ax1, ax2]]
[ax.plot(good_point[:, 1], good_point[:, 0], 'go') for ax in [ax1, ax2]]
ax1.imshow(map, cmap='gray')
ax2.imshow(map, cmap='gray')
plt.show()