import copy
import threading
from enum import Enum

import numpy as np

from typing import Union, Tuple

class FrameType(Enum):
    """The type of a frame."""
    WORLD = 0
    LOCAL = 1
    UNKNOWN = 2

class Frame:
    """A frame is a 2D orthogonal coordinate system with an origin and an angle."""
    def __init__(self, origin:Tuple[float]=(0, 0), angle:float=0) -> None:
        self.x = origin[0]
        self.y = origin[1]
        self.angle = angle

    def frame_type(self, unknown:bool=False) -> FrameType:
        """Return the type of the frame."""
        if unknown:
            return FrameType.UNKNOWN
        if self.x == 0 and self.y == 0 and self.angle == 0:
            return FrameType.WORLD
        else:
            return FrameType.LOCAL

class WorldFrame(Frame):
    """The world frame is a singleton."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs): # There should be only one world frame!
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(WorldFrame, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        super().__init__()


class Transform:
    def __init__(self) -> None:
        pass

    @staticmethod
    def affine_transform(state: Union[list, np.ndarray], rotation: float, translation:tuple=None, scale:float=1) -> np.ndarray:
        """Return the transformed state.
        Args:
            state: The state to be transformed.
            rotation: The rotation angle in degrees.
            translation: The translation vector.
            scale: The scale factor.
        """
        tr_state = np.array(state).reshape(-1, 1)
        t = np.array(translation) if translation is not None else np.array([0, 0])
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        tr_state[:2] = scale * R @ tr_state[:2] + t
        return tr_state
    
    @staticmethod
    def frame2frame_transform(state: Union[list, np.ndarray], src_frame: Frame, dst_frame: Frame) -> np.ndarray:
        """Return the transformed state.
        Args:
            state: The state to be transformed.
            src_frame: The source frame.
            dst_frame: The destination frame.
        """
        tr_state = np.array(state).reshape(-1, 1)
        rotation = dst_frame.angle - src_frame.angle
        translation = (dst_frame.x-src_frame.x, dst_frame.y-src_frame.y)
        tr_state = Transform.affine_transform(tr_state, rotation, translation)
        return tr_state


class ScaleOffsetReverseTransform:
    def __init__(self, scale:float=1, offsetx_after:float=0, offsety_after:float=0, x_reverse=False, y_reverse=False, x_max_before=0, y_max_before=0):
        """Transform the given coordinates by some scales and offsets.
        Args:
            scale       : Scale factor.
            offset_after: The offset of x and y axes.
            reverse     : If the x and y axes should be reversed.
            max_before  : The maximal values along x and y axes, used to calculate the reversed coordinates.
        Comments:
            For orginal coordinates z=[x,y], if x or y is reversed, calculate the reversed coordinate first.
            Then calculate the transformed coordinates according to the scaling and the offset.
        """
        self.k = [scale, scale]
        self.b = [offsetx_after, offsety_after]
        self.xr = x_reverse
        self.yr = y_reverse
        self.xm = x_max_before
        self.ym = y_max_before

    def __call__(self, state:Union[list, np.ndarray], forward=True) -> Union[list, np.ndarray]:
        """Return the transformed state. If forward=False, it means from the transformed state to the original one.
        """
        if isinstance(state, tuple):
            tr_state = list(state)
        else:
            tr_state = copy.copy(state)
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
        '''Return transformed/original coordinates, in shape (2*n).
        '''
        cvt_x = self.cvt_coord_x(x, forward)
        cvt_y = self.cvt_coord_y(y, forward)
        return np.hstack((cvt_x[:,np.newaxis], cvt_y[:,np.newaxis]))

