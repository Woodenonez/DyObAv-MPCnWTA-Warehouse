from typing import Tuple

import numpy as np

from PIL import Image, ImageFilter
from skimage import morphology, filters

import matplotlib.pyplot as plt

def np_sigmoid(x):
    return 1/(1 + np.exp(-x))

def np_create_circle(r, ring=False):
    # Generate a n-by-n matrix, n=2r+1
    # This matrix contains a circle filled with 1, otherwise 0
    # If 'ring' is true, only the edge of the circle is 1.
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    if ring:
        return (np.abs(dists-r)<0.5).astype(int)
    else:
        return (dists<r).astype(int)

def np_create_circle_mask(circle_centre, r, base_matrix):
    # Create a circle mask on an empty matrix with the given shape
    # If the maskis out of borders, it will be cut.
    base_matrix = np.zeros(base_matrix.shape)
    np_circle = np_create_circle(r)
    row_min = np.maximum(circle_centre[1]-r, 0)
    row_max = np.minimum(circle_centre[1]+r, base_matrix.shape[0]-1)
    col_min = np.maximum(circle_centre[0]-r, 0)
    col_max = np.minimum(circle_centre[0]+r, base_matrix.shape[1]-1)
    if row_max == base_matrix.shape[0]-1:
        if -(r-(base_matrix.shape[0]-1-circle_centre[1])) != 0:
            np_circle = np_circle[:-(r-(base_matrix.shape[0]-1-circle_centre[1])),:]
    if row_min == 0:
        np_circle = np_circle[r-circle_centre[1]:,:]
    if col_max == base_matrix.shape[1]-1:
        if -(r-(base_matrix.shape[1]-1-circle_centre[0])) != 0:
            np_circle = np_circle[:, :-(r-(base_matrix.shape[1]-1-circle_centre[0]))]
    if col_min == 0:
        np_circle = np_circle[:, r-circle_centre[0]:]
    base_matrix[row_min:row_max+1, col_min:col_max+1] = np_circle
    return base_matrix

def np_add_circle_mask(circle_centre, r, base_matrix):
    # Add a circle mask to the given base
    mask = np_create_circle_mask(circle_centre, r, base_matrix)
    base_matrix += mask
    return base_matrix

def np_matrix_subtract(base_matrix, ref_matrix):
    # Subtract the intersection area of the base with the reference 
    intersection = base_matrix.astype(int) & ref_matrix.astype(int)
    base_matrix -= intersection
    return base_matrix

def np_dist_map(centre, base_matrix):
    # Create a distance map given the centre and map size
    # The distance is defined as the Euclidean distance
    # The map is normalized
    base_matrix = np.zeros(base_matrix.shape)
    x = np.arange(0, base_matrix.shape[1])
    y = np.arange(0, base_matrix.shape[0])
    x, y = np.meshgrid(x, y)
    base_matrix = np.linalg.norm(np.stack((x-centre[0], y-centre[1])), axis=0)
    return base_matrix/base_matrix.max()

def np_gaussian(xy:tuple, mu:tuple, sigma:tuple, rho): #XXX TEST FUNCTION
    in_exp = -1/(2*(1-rho**2)) * ((xy[0]-mu[0])**2/(sigma[0]**2) 
                                + (xy[1]-mu[1])**2/(sigma[1]**2) 
                                - 2*rho*(xy[0]-mu[0])/(sigma[0])*(xy[1]-mu[1])/(sigma[1]))
    z = 1/(2*np.pi*sigma[0]*sigma[1]*np.sqrt(1-rho**2)) * np.exp(in_exp)
    return z

def np_gaudist_map(centre, base_matrix, sigmas=[100,100], rho=0, flip=False):
    # Create a distance map given the centre and map size
    # The distance is defined by the Gaussian distribution
    # The map is normalized
    sigma_x, sigma_y = sigmas[0], sigmas[1]
    x = np.arange(0, base_matrix.shape[1])
    y = np.arange(0, base_matrix.shape[0])
    x, y = np.meshgrid(x, y)
    in_exp = -1/(2*(1-rho**2)) * ((x-centre[0])**2/(sigma_x**2) 
                                + (y-centre[1])**2/(sigma_y**2) 
                                - 2*rho*(x-centre[0])/(sigma_x)*(y-centre[1])/(sigma_y))
    z = 1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)) * np.exp(in_exp)
    if flip:
        return 1 - z/z.max()
    else:
        return z/z.max()

def np_multigau_map(gaumap_list, weights=[]):
    if not weights:
        weights = [1/(x+1) for x in range(len(gaumap_list))]
    out = weights[0]*gaumap_list[0]
    for i in range(len(gaumap_list)-1):
        out += weights[i+1]*gaumap_list[i+1]
        out[out>1] = 1
    return out/out.max()

def __divid_inout_points(points:np.ndarray, occupancy_map:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = points.astype(int)
    occupied_idc = occupancy_map[coords[:,1], coords[:,0]] > 0
    points_in  = points[occupied_idc]
    points_out = points[~occupied_idc]
    return points_in, points_out

def get_closest_edge_point(original_points:np.ndarray, occupancy_map:np.ndarray) -> np.ndarray:
    if len(original_points.shape) == 1:
        original_points = original_points[np.newaxis,:]
    if len(occupancy_map.shape) != 2:
        raise ValueError(f'Input map should be 2-dim (got {len(occupancy_map.shape)}-dim).')
    if original_points.shape[1] != 2:
        raise ValueError(f'Input point shape incorrect (should be 2, got {original_points.shape[1]}).')
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

    points_in, points_out = __divid_inout_points(original_points, occupancy_map)
    if points_in.shape[0] == 0:
        return points_out
        
    edge_map = filters.roberts(morphology.dilation(occupancy_map, np.ones((3,3))))
    edge_map[edge_map>0] = 1
    dist_map = np_dist_map(points_in, occupancy_map) * edge_map[:,:,np.newaxis].repeat(points_in.shape[0], axis=2)
    dist_map[dist_map==0] = np.max(dist_map)
    closest_edge_point = []
    for i in range(points_in.shape[0]):
        closest_edge_point.append(np.unravel_index(np.argmin(dist_map[:,:,i], axis=None), dist_map[:,:,i].shape))
    processed_points = np.vstack((np.array(closest_edge_point)[:,::-1], points_out))
    return processed_points

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    centre1 = (30,40)
    centre2 = (60,60)
    centre3 = (80,80)
    centres = [centre1, centre2, centre3]

    X = np.arange(0, 120)
    Y = np.arange(0, 100)
    X, Y = np.meshgrid(X, Y)
    base = np.zeros((100,120))

    map_d = np_dist_map(centre1, base)

    map1 = np_gaudist_map(centre1, base, sigmas=[10,10], rho=0, flip=False)
    map2 = np_gaudist_map(centre2, base, sigmas=[10,10], rho=0, flip=False)
    map3 = np_gaudist_map(centre3, base, sigmas=[10,10], rho=0, flip=False)

    map = map1 + map2/2 + map3/3
    map = map/np.max(map)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, map, cmap='hot')

    plt.imshow(map1)
    [plt.plot(c[0],c[1],'rx') for c in centres]

    plt.show()