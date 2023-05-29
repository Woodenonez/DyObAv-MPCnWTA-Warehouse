import numpy as np
import matplotlib.pyplot as plt

import pyclipper

from typing import List, Tuple, Callable
from matplotlib.axes import Axes


class Inflator:
    def __init__(self) -> None:
        self.inflator = pyclipper.PyclipperOffset()

    def inflate_polygon(self, polygon:List[list], inflate_margin:float):
        """
        Args:
            polygon: Vertex (2D) list of the original polygon.
            inflate_margin: The margin to be inflated, if negative then deflate.
        Returns:
            inflated_polygon: Vertex (2D) list of the inflated polygon.
        """
        self.inflator.Clear()
        self.inflator.AddPath(pyclipper.scale_to_clipper(polygon), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_polygon = pyclipper.scale_from_clipper(self.inflator.Execute(pyclipper.scale_to_clipper(inflate_margin)))[0]
        return inflated_polygon

    def inflate_polygons(self, polygons:List[List[list]], inflate_margin:float):
        inflated_polygons = []
        for poly in polygons:
            inflated_poly = self.inflate_polygon(poly, inflate_margin)
            inflated_polygons.append(inflated_poly)
        return inflated_polygons


class GeometricMap:
    """With boundary and obstacle coordinates."""
    def __init__(self, boundary_coords:List[tuple], obstacle_list:List[List[tuple]], inflate_margin:float=None):
        """
        Args:
            boundary_coords: A list of tuples, each tuple is a pair of coordinates.
            obstacle_list: A list of lists of tuples, each tuple is a pair of coordinates.
            inflator: A function that inflates a polygon.
        """
        boundary_coords, obstacle_list = self.__input_validation(boundary_coords, obstacle_list)
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.inflator = Inflator()
        if inflate_margin is not None:
            self.processed_boundary_coords = self.inflator.inflate_polygon(boundary_coords, -inflate_margin)
            self.processed_obstacle_list = self.inflator.inflate_polygons(obstacle_list, inflate_margin)
        else:
            self.processed_boundary_coords = None
            self.processed_obstacle_list = None

    def __input_validation(self, boundary_coords, obstacle_list):
        if not isinstance(boundary_coords, list):
            raise TypeError('A map boundary must be a list of tuples.')
        if not isinstance(obstacle_list, list):
            raise TypeError('A map obstacle list must be a list of lists of tuples.')
        if len(boundary_coords[0])!=2 or len(obstacle_list[0][0])!=2:
            raise TypeError('All coordinates must be 2-dimension.')
        return boundary_coords, obstacle_list

    def __call__(self, inflated:bool=True) -> Tuple[List[tuple], List[List[tuple]]]:
        if inflated:
            if self.processed_boundary_coords is None or self.processed_obstacle_list is None:
                raise ValueError('No inflated map available.')
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list
    
    def coords_cvt(self, ct:Callable) -> 'GeometricMap':
        self.boundary_coords = [ct(x) for x in self.boundary_coords]
        self.obstacle_list = [[ct(x) for x in y] for y in self.obstacle_list]
        self.processed_boundary_coords = [ct(x) for x in self.processed_boundary_coords]
        self.processed_obstacle_list = [[ct(x) for x in y] for y in self.processed_obstacle_list]
        return self

    def get_occupancy_map(self, rescale:int=100) -> np.ndarray:
        """
        Args:
            rescale: The resolution of the occupancy map.
        Returns:
            A numpy array of shape (height, width, 3).
        """
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
        return occupancy_map

    def plot(self, ax: Axes, inflated:bool=True, original_plot_args:dict={'c':'k'}, inflated_plot_args:dict={'c':'r', 'alpha':0.3}):
        if inflated:
            if self.processed_boundary_coords is None or self.processed_obstacle_list is None:
                raise ValueError('No inflated map available.')
            else:
                plot_boundary = np.array(self.processed_boundary_coords+[self.processed_boundary_coords[0]])
                ax.plot(plot_boundary[:,0], plot_boundary[:,1], **inflated_plot_args)
                for coords in self.processed_obstacle_list:
                    plot_obstacle = np.array(coords+[coords[0]])
                    plt.fill(plot_obstacle[:,0], plot_obstacle[:,1], **inflated_plot_args)
        plot_boundary = np.array(self.boundary_coords+[self.boundary_coords[0]])
        if original_plot_args is not None:
            ax.plot(plot_boundary[:,0], plot_boundary[:,1], **original_plot_args)
            for coords in self.obstacle_list:
                plot_obstacle = np.array(coords+[coords[0]])
                plt.fill(plot_obstacle[:,0], plot_obstacle[:,1], **original_plot_args)


if __name__ == '__main__':
    boundary = [(0,0), (10,0), (10,10), (0,10)]
    obstacle_list = [[(1,1), (2,1), (2,2), (1,2)], [(3,3), (4,3), (4,4), (3,4)]]
    map = GeometricMap(boundary, obstacle_list, 0.2)
    map.get_occupancy_map()
    fig, ax = plt.subplots()
    map.plot(ax, inflated=False)
    plt.show()