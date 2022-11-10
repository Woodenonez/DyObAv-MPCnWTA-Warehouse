import math
import random

from warnings import warn
from typing import List, Tuple

import pyclipper # for geometric map inflation
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from scipy import spatial
from skimage import util, color, filters, measure, morphology

from util.basic_datatype import *

class CoordTransform:
    def __init__(self, scale:float=1, offsetx_after:float=0, offsety_after:float=0, x_reverse=False, y_reverse=False, x_max_before=0, y_max_before=0):
        self.k = [scale, scale]
        self.b = [offsetx_after, offsety_after]
        self.xr = x_reverse
        self.yr = y_reverse
        self.xm = x_max_before
        self.ym = y_max_before

    def __call__(self, state:NumpyState, forward=True) -> NumpyState:
        tr_state = state.copy()
        if forward:
            if self.xr:
                tr_state[0] = self.xm - tr_state[0]
            if self.yr:
                tr_state[1] = self.ym - tr_state[1]
            tr_state[0] = tr_state[0]*self.k[0]+self.b[0]
            tr_state[1] = tr_state[1]*self.k[1]+self.b[1]
        else:
            tr_state[0] = (state[0]-self.b[0]) / self.k[0]
            tr_state[1] = (state[1]-self.b[1]) / self.k[1]
            if self.xr:
                tr_state[0] = self.xm - tr_state[0]
            if self.yr:
                tr_state[1] = self.ym - tr_state[1]
        return tr_state

    def cvt_coord_x(self, x:np.ndarray, forward=True) -> np.ndarray:
        if forward:
            if self.xr:
                x = self.xm - x
            cvt_x = self.k[0]*x + self.b[0]
        else:
            cvt_x = (x-self.b[0]) / self.k[0]
            if self.xr:
                cvt_x = self.xm - cvt_x
        return cvt_x

    def cvt_coord_y(self, y:np.ndarray, forward=True) -> np.ndarray:
        if forward:
            if self.yr:
                y = self.ym - y
            cvt_y = self.k[1]*y + self.b[1]
        else:
            cvt_y = (y-self.b[1]) / self.k[1]
            if self.yr:
                cvt_y = self.ym - cvt_y
        return cvt_y

    def cvt_coords(self, x:np.ndarray, y:np.ndarray, forward=True) -> np.ndarray:
        cvt_x = self.cvt_coord_x(x, forward)
        cvt_y = self.cvt_coord_y(y, forward)
        return np.hstack((cvt_x[:,np.newaxis], cvt_y[:,np.newaxis]))

class OccupancyMap:
    def __init__(self, map_image:np.ndarray, occupancy_threshold:int=120):
        map_image = self.__input_validation(map_image)
        self.width = map_image.shape[1]
        self.height = map_image.shape[0]

        self.__background = map_image
        self.__grayground = color.rgb2gray(map_image) if map_image.shape[2]==3 else map_image[:,:,0]
        self.__binyground = self.__get_occupancy_map(threshold=occupancy_threshold)

    def __input_validation(self, map_image):
        if not isinstance(map_image, np.ndarray):
            raise TypeError('A map image must be a numpy array.')
        if len(map_image.shape) == 2: # add channel dimension
            map_image = map_image[:, :, np.newaxis]
        if len(map_image.shape) != 3:
            raise TypeError(f'A map image must have 2 or 3 dimensions; Got {len(map_image.shape)}.')
        if map_image.shape[2]==4: # the 4th channel will be discarded
            map_image = map_image[:, :, :3]
        if map_image.shape[2] not in [1, 3]: # the 4th channel can be alpha-channel
            raise TypeError(f'A map image must have 1/3/4 channels; Got {map_image.shape[2]}.')
        return map_image

    def __get_occupancy_map(self, threshold) -> np.ndarray:
        threshold = min(255, max(0, threshold))
        return self.__grayground>threshold

    def __call__(self, binary_scale=False, gray_scale=True) -> np.ndarray:
        if binary_scale:
            return self.__binyground
        if gray_scale:
            return self.__grayground
        return self.__background

    def __getitem__(self, pixel_positions, gray_scale=True) -> np.ndarray:
        if gray_scale:
            return self.__grayground[pixel_positions]
        return self.__background[pixel_positions]
            
    def get_edge_map(self, dilation_size:int=3):
        if dilation_size > 0:
            edge_map = filters.roberts(morphology.dilation(self.__binyground, np.ones((dilation_size, dilation_size))))
        else:
            edge_map = filters.roberts(self.__binyground)
        return util.invert(edge_map)

    def get_geometric_map(self, inflation=0):
        boundary_coords = [(0,0), (0,self.height), (self.width, self.height), (self.width, 0)]
        obstacle_list = []
        for contour in measure.find_contours(self.__grayground):
            hull_points = contour[spatial.ConvexHull(contour).vertices, :]
            coords = self.minimum_bounding_rectangle(hull_points)
            # coords = measure.approximate_polygon(contour, tolerance=5)
            obstacle_list.append(coords[:, ::-1]) # contour need x-y swap
        for coords in obstacle_list[::-1]:
            x1_left  = min(coords[:,0])
            x1_right = max(coords[:,0])
            y1_low = min(coords[:,1])
            y1_up  = max(coords[:,1])
            for other_coords in obstacle_list:
                sorted_coords_x = np.sort(other_coords[:,0])
                sorted_coords_y = np.sort(other_coords[:,1])
                x2_left  = sorted_coords_x[1]
                x2_right = sorted_coords_x[2]
                y2_low = sorted_coords_y[1]
                y2_up  = sorted_coords_y[2]
                if (x1_left>x2_left) & (x1_right<x2_right) & (y1_low>y2_low) & (y1_up<y2_up):
                    obstacle_list.pop(-1)
                    continue
        obstacle_list = [x.tolist() for x in obstacle_list]
        return GeometricMap(boundary_coords, obstacle_list, inflation)

    def minimum_bounding_rectangle(self, hull_points):
        """
        https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
        Find the smallest bounding rectangle for a convex hull.
        Returns a set of points representing the corners of the bounding box.

        :param points: an n*2 matrix of coordinates
        :rval: an n*2 matrix of coordinates
        """
        # calculate edge angles
        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, np.pi/2))
        angles = np.unique(angles)

        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-np.pi/2),
            np.cos(angles+np.pi/2),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval

class GeometricMap:
    def __init__(self, boundary:List[tuple], obstacle_list:List[List[tuple]], inflation=0):
        boundary, obstacle_list = self.__input_validation(boundary, obstacle_list)
        self.boundary_coords = boundary
        self.obstacle_list = obstacle_list
        self.inflation(inflation) #  get processed map

    def __input_validation(self, boundary, obstacle_list):
        if not isinstance(boundary, list):
            raise TypeError('A map boundary must be a list of tuples.')
        if not isinstance(obstacle_list, list):
            raise TypeError('A map obstacle list must be a list of lists of tuples.')
        if len(boundary[0])!=2 or len(obstacle_list[0][0])!=2:
            raise TypeError('All coordinates must be 2-dimension.')
        return boundary, obstacle_list

    def __call__(self, inflated:bool=True) -> Tuple[List[tuple], List[List[tuple]]]:
        if inflated:
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list

    def __preprocess_obstacle(self, obstacle, inflation):
        self.inflator.Clear()
        self.inflator.AddPath(obstacle, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_obstacle = pyclipper.scale_from_clipper(self.inflator.Execute(inflation))[0]
        return inflated_obstacle    
    
    def __preprocess_obstacles(self, obstacle_list, inflation):
        inflated_obstacles = []
        for obs in obstacle_list:
            obstacle = pyclipper.scale_to_clipper(obs)
            inflated_obstacle = self.__preprocess_obstacle(obstacle, inflation)
            inflated_obstacle.reverse() # obstacles are ordered clockwise
            inflated_obstacles.append(inflated_obstacle)
        return inflated_obstacles

    def inflation(self, inflate_margin):
        self.inflator = pyclipper.PyclipperOffset()
        self.processed_obstacle_list   = self.__preprocess_obstacles(self.obstacle_list, 
                                                                     pyclipper.scale_to_clipper(inflate_margin))
        self.processed_boundary_coords = self.__preprocess_obstacle( pyclipper.scale_to_clipper(self.boundary_coords), 
                                                                     pyclipper.scale_to_clipper(-inflate_margin))

    def rescale(self, rescale:float):
        self.boundary_coords = [tuple(np.array(x)*rescale) for x in self.boundary_coords]
        self.obstacle_list = [[tuple(np.array(x)*rescale) for x in y] for y in self.obstacle_list]
        self.processed_boundary_coords = [tuple(np.array(x)*rescale) for x in self.processed_boundary_coords]
        self.processed_obstacle_list = [[tuple(np.array(x)*rescale) for x in y] for y in self.processed_obstacle_list]

    def coords_cvt(self, ct:CoordTransform):
        self.boundary_coords = [tuple(ct(np.array(x))) for x in self.boundary_coords]
        self.obstacle_list = [[tuple(ct(np.array(x))) for x in y] for y in self.obstacle_list]
        self.processed_boundary_coords = [tuple(ct(np.array(x))) for x in self.processed_boundary_coords]
        self.processed_obstacle_list = [[tuple(ct(np.array(x))) for x in y] for y in self.processed_obstacle_list]

    def get_occupancy_map(self, rescale:int=100, threshold=120) -> OccupancyMap:
        if not isinstance(rescale, int):
            raise TypeError(f'Rescale factor must be int, got {type(rescale)}.')
        assert(0<rescale<2000),(f'Rescale value {rescale} is abnormal.')
        boundary_np = np.array(self.boundary_coords)
        width  = max(boundary_np[:,0]) - min(boundary_np[:,0])
        height = max(boundary_np[:,1]) - min(boundary_np[:,1])
        fig, ax = plt.subplots(figsize=(width, height), dpi=rescale)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.plot(np.array(self.boundary_coords)[:,0], np.array(self.boundary_coords)[:,1], 'w-')
        for coords in self.obstacle_list:
            x, y = np.array(coords)[:,0], np.array(coords)[:,1]
            plt.fill(x, y, color='k')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        occupancy_map = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        occupancy_map = occupancy_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return OccupancyMap(occupancy_map, threshold)


class Node:
    def __init__(self, x, y=None, theta=None, id=-1):
        if isinstance(x, (list, tuple)):
            self.x, self.y = x[0], x[1]
            if len(x) < 3:
                self.theta = 0
            else:
                self.theta = x[2]
            self.id = id
        elif y is None:
            raise ValueError('Y-axis value cannot be None.')
        else:
            self.x = x
            self.y = y
            self.theta = theta
            self.id = id # ID=-1 means no ID.

    def __call__(self) -> tuple:
        return (self.x, self.y, self.theta, self.id)

    def __getitem__(self, var_idx) -> float:
        # warn('Better not to use position index to access a Node.')
        return self.__call__()[var_idx]

    def __sub__(self, other_node):
        return math.hypot(self.x-other_node.x, self.y-other_node.y)

    def rescale(self, rescale):
        self.x = self.x*rescale
        self.y = self.y*rescale

class Path(object):
    '''
    Two indicators:
    1. Node ID
    2. Node position index
    '''
    def __init__(self, path:List[Node]) -> None:
        self.__path = self.__input_validation(path)
        self.__path_id_dict = self.__cvt2dict()

    def __input_validation(self, path):
        if not isinstance(path, list):
            raise TypeError('A path must be a list.')
        if path:
            if not isinstance(path[0], Node):
                raise TypeError('All path elements must be Nodes.')
        # else:
        #     warn('The input path is empty.')
        return path

    def __cvt2dict(self) -> dict:
        path_dict = {}
        for node in self.__path:
            path_dict[node.id] = (node.x, node.y)
        return path_dict
    
    def __len__(self):
        return len(self.__path)

    def __call__(self, to_tuple=True):
        if to_tuple:
            return [x() for x in self.__path]
        return self.__path

    def __getitem__(self, node_position:int) -> Node:
        # return based on node position
        return self.__path[node_position]

    def append(self, node:Node) -> None:
        if not isinstance(node, Node):
            raise TypeError(f'Input must be a Node, got {type(node)}.')
        self.__path.append(node)

    def insert(self, node_position:int, node:Node) -> None:
        if not isinstance(node, Node):
            raise TypeError(f'Input must be a Node, got {type(node)}.')
        self.__path.insert(node_position, node)

    def pop(self, node_position:int=-1) -> Node:
        return self.__path.pop(node_position)

    def remove(self, node:Node) -> None:
        self.__path.remove(node)

    def get_node_coords(self, node_id)-> tuple:
        # return based on node id
        return self.__path_id_dict[node_id]

    def rescale(self, rescale):
        [n.rescale(rescale) for n in self.__path]
        self.__path_id_dict = self.__cvt2dict()

class Trajectory(object):
    def __init__(self, trajectory:List[NumpyState], base_path:Path=None):
        self.traj = trajectory
        self.base_path = base_path

    def __len__(self):
        return len(self.traj)

    def __call__(self):
        return self.traj

    def __getitem__(self, position:int) -> NumpyState:
        # return based on node position
        return self.traj[position]

    def append(self, state:NumpyState) -> None:
        self.traj.append(state)

    def pop(self, position:int=-1) -> Node:
        return self.traj.pop(position)

    def remove(self, state:NumpyState) -> None:
        self.traj.remove(state)

    def get_base_path_node(self, node_position:int):
        return self.base_path[node_position]

    def update_base_path(self, new_bath_path:Path):
        self.base_path = new_bath_path


class NetGraph(object):
    def __init__(self, nxgraph:nx.Graph):
        self.G = nxgraph
        self.nodes = self.G.nodes
        self.edges = self.G.edges

    def __getitem__(self, node_id):
        return self.G[node_id]

    def get_x(self, node_id):
        return self.G.nodes[node_id]['pos'][0]

    def get_y(self, node_id):
        return self.G.nodes[node_id]['pos'][1]

    def get_node(self, node_id):
        return Node(self.get_x(node_id), self.get_y(node_id), node_id)

    def get_node_pos(self, node_id):
        return self.get_x(node_id), self.get_y(node_id)

    def get_all_edge_positions(self) -> List[List[tuple]]:
        edge_positions = []
        for e in self.edges:
            edge_positions.append([self.get_node_pos(e[0]), self.get_node_pos(e[1])])
        return edge_positions

    def return_given_path(self, path_node_idc:list) -> Path:
        path = Path([self.get_node(idx) for idx in path_node_idc])
        return path

    def return_random_path(self, start_node_index, num_traversed_nodes:int) -> Path:
        # return random path without repeat nodes
        path_idx = [start_node_index]
        path = Path([self.get_node(start_node_index)])
        for _ in range(num_traversed_nodes):
            connected_nodes_idx = list(self.G.adj[path_idx[-1]])
            connected_nodes_idx = [x for x in connected_nodes_idx if x not in path_idx]
            if not connected_nodes_idx:
                return path
            next_idx = random.choice(connected_nodes_idx) # NOTE: Change this to get desired path pattern
            path_idx.append(next_idx)
            path.append(self.get_node(next_idx))
        return path


if __name__ == '__main__':
    node_1 = Node(2,3,0)
    node_11 = Node(3,4,1)
    node_12 = Node(4,4,1)
    node_2 = Node(5,4,1)
    path_list = [node_1, node_2]
    traj_list = [node_1, node_11, node_2]
    path = Path(path_list)
    traj = Trajectory(traj_list, base_path=path)
    print(path(), path[1](), path.remove_node(0)(), path())
    print(traj(), traj[1](), traj[1:])