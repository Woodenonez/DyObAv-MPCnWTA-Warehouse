import os
import json
import pathlib
import numpy as np

from basic_map.graph_basic import NetGraph
from basic_map.map_geometric import GeometricMap
from basic_map.map_occupancy import OccupancyMap

from io import BufferedReader
from typing import List

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]


class MapInterface:
    """Interface for map classes.
    
    Receive external data and construct a map object.
    """
    def __init__(self, raw_map_dir: str) -> None:
        self._prt_name = 'MapInterface'
        self.raw_map_dir = os.path.join(ROOT_DIR, 'data', raw_map_dir)

    def cvt_occ2geo(self, occ_map: OccupancyMap, inflate_margin: float) -> GeometricMap:
        """Convert an occupancy map to a geometric map."""
        geo_map_input = occ_map.get_geometric_map()
        geo_map = GeometricMap(*geo_map_input, inflate_margin=inflate_margin)
        return geo_map

    def get_occ_map_from_pgm(self, pgm_fname, occupancy_threshold: int, inversed_pixel:bool=False, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> OccupancyMap:
        """Read a map from a PGM file and return the map as a numpy array."""
        with open(os.path.join(self.raw_map_dir, pgm_fname), 'rb') as pgmf:
            the_map = self.read_pgm_and_process(pgmf, inversed_pixel, bit_depth, one_line_head, skip_second_line)
        occ_map = OccupancyMap(np.array(the_map), occupancy_threshold)
        return occ_map
    
    def get_graph_from_json(self, json_fname) -> NetGraph:
        """Read a graph from a JSON file and return the graph as a NetGraph object."""
        graph = NetGraph.load_from_json(os.path.join(self.raw_map_dir, json_fname))
        return graph

    @staticmethod
    def read_pgm(pgmf:BufferedReader, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> List[list]:
        """Return a raster of integers from a PGM file as a list of lists (The head is normally [P5 Width Height Depth])."""
        header = pgmf.readline()  # the 1st line
        if one_line_head:
            magic_num = header[:2]
            (width, height) = [int(i) for i in header.split()[1:3]]
            depth = int(header.split()[3])
        else:
            magic_num = header
            if skip_second_line:
                comment = pgmf.readline() # the 2nd line if there is
                print(f'Comment: [{comment}]')
            (width, height) = [int(i) for i in pgmf.readline().split()]
            depth = int(pgmf.readline())

        if bit_depth == 8:
            assert magic_num[:2] == 'P5'
            assert depth <= 255
        elif bit_depth == 16:
            assert magic_num[:2] == b'P5'
            assert depth <= 65535

        raster = []
        for _ in range(height):
            row = []
            for _ in range(width):
                row.append(ord(pgmf.read(1)))
            raster.append(row)
        return raster
    
    @staticmethod
    def read_pgm_and_process(pgmf:BufferedReader, inversed_pixel:bool, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True) -> np.ndarray:
        raw_map = MapInterface.read_pgm(pgmf, bit_depth, one_line_head, skip_second_line)
        the_map = np.array(raw_map)
        if inversed_pixel:
            the_map = 255 - the_map
        the_map[the_map>10] = 255
        the_map[the_map<=10] = 0
        the_map[:,[0,-1]] = 0
        the_map[[0,-1],:] = 0
        return the_map



