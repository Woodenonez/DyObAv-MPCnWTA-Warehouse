import casadi.casadi as cs
from .mpc_helper import *

from typing import Union, List

def cost_inside_cvx_polygon(point: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX, weight:Union[cs.SX, float]=1.0):
    """Cost (weighted squared) for being inside a convex polygon defined by `b - [a0,a1]*[x,y]' > 0`.
        
    Arguments:
        point: The (1*n)-dim target point.
        b: Shape  (1*m) with m half-space offsets.
        a0: Shape (1*m) with m half-space weight vectors.
        a1: Shape (1*m) with m half-space weight vectors.
        
    Returns:
        is_inside: The (1*1)-dim indicator. If inside, return positive value, else return 0.

    Comments:
        Each half-space if defined as `b - [a0,a1]*[x,y]' > 0`.
        If prod(|max(0,all)|)>0, then the point is inside; Otherwise not.
    """
    indicator = inside_cvx_polygon(point, b, a0, a1) # indicator<0, if outside pollygon
    cost = weight * indicator**2
    return cost

def cost_inside_ellipses(point: cs.SX, ellipse_param: List[cs.SX], weight:Union[cs.SX, float]=1.0):
    """Cost (weighted squared) for being inside a set of ellipses defined by `(cx, cy, sx, sy, angle, alpha)`.
    
    Arguments:
        point: The (1*n)-dim target point.
        ellipse_param: Shape (5 or 6 * m) with m ellipses. 
                       Each ellipse is defined by (cx, cy, rx, ry, angle, alpha).
                       
    Returns:
        is_inside: The (1*m)-dim indicator vector. If inside, return positive value, else return negative value.
    """
    if len(ellipse_param) > 5:
        alpha = ellipse_param[5]
    else:
        alpha = 1
    indicator = inside_ellipses(point, ellipse_param) # indicator<0, if outside ellipse
    indicator = weight * alpha * cs.fmax(0.0, indicator)**2
    cost = cs.sum1(indicator)
    return cost

def cost_control_actions(actions: cs.SX, weights:Union[List[cs.SX], float]=1.0):
    """Cost (weighted squared) for control action.
    
    Arguments:
        actions: The (1*n)-dim control action.
    """
    cost = cs.sum2(weights * actions**2) # row-wise sum
    return cost

def cost_control_jerks(actions: cs.SX, last_actions: cs.SX, weights:Union[List[cs.SX], float]=1.0):
    """Cost (weighted squared) for control jerk.

    Arguments:
        actions: The (1*n)-dim control action.
        last_actions: The (1*n)-dim control action at last time step.
    """
    cost = cs.sum2(weights * (actions-last_actions)**2)
    return cost

def cost_fleet_collision(point: cs.SX, points: cs.SX, safe_distance: float, weight:Union[cs.SX, float]=1.0):
    """Cost (weighted squared) for colliding with other robots.
    
    Arguments:
        point: The (1*n)-dim target point.
        points: The (m*n)-dim points of other robots.
        
    Comments:
        Only have cost when the distance is smaller than `safe_distance`.
    """
    cost = weight * cs.sum2(cs.fmax(0.0, safe_distance**2 - dist_to_points_square(point, points)))
    return cost

def cost_refvalue_deviation(actual_value: cs.SX, ref_value: cs.SX, weight:float=1.0):
    return weight * (actual_value-ref_value)**2

def cost_refstate_deviation(state: cs.SX, ref_state: cs.SX, weights:Union[List[cs.SX], float]=1.0):
    return weights * cs.sum2((state-ref_state)**2)

def cost_refpath_deviation(point: cs.SX, line_segments: cs.SX, weight:Union[cs.SX, float]=1.0):
    """Reference deviation cost (weighted squared) penalizes on the deviation from the reference path.
    
    Arguments:
        line_segments: The (m*n)-dim var with m n-dim points.
    """
    distances_sqrt = cs.SX.ones(1)
    for i in range(line_segments.shape[0]-1):
        distance = dist_to_lineseg(point[:2], line_segments[i:i+2,:2])
        distances_sqrt = cs.horzcat(distances_sqrt, distance**2)
    cost = cs.mmin(distances_sqrt[1:]) * weight
    return cost

def cost_refpoint_detach(point: cs.SX, ref_point: cs.SX, ref_distance: Union[cs.SX, float], weight:Union[cs.SX, float]=1.0):
    """Reference detachment cost (weighted squared) penalizes on the deviation from a certain range of the reference point.
    
    Arguments:
        ref_point: The (1*n)-dim var with n-dim point.
    Comments:
        The robot should stay a constant distance with the reference point.
    """
    actual_distance = cs.sqrt(cs.sum2((point-ref_point)**2))
    cost = (actual_distance - ref_distance)**2 * weight
    return cost

