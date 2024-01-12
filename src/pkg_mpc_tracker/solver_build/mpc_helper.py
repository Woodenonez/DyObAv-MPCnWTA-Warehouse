import casadi.casadi as cs

from typing import List

def dist_to_points_square(point: cs.SX, points: cs.SX):
    """Calculate the squared distance from a target point to a set of points.
    
    Arguments:
        point: The (1*n)-dim target point.
        points: Shape (m*n) with m points.

    Returns:
        The (1*m)-dim squared distance from the target point to each point in the set.
    """
    return cs.sum1((point.T-points.T)**2) # sum1 is summing each column

def dist_to_lineseg(point: cs.SX, line_segment: cs.SX):
    """Calculate the distance from a target point to a line segment.

    Arguments:
        point: The (1*n)-dim target point.
        line_segment: The (2*n) edge points.

    Returns:
        distance: The (1*1)-dim distance from the target point to the line segment.

    Reference:
        Link: https://math.stackexchange.com/questions/330269/the-distance-from-a-point-to-a-line-segment
    """
    (p, s1, s2) = (point[:2], line_segment[[0],:], line_segment[[1],:])
    s2s1 = s2-s1 # line segment
    t_hat = cs.dot(p-s1,s2s1)/(s2s1[0]**2+s2s1[1]**2+1e-16)
    t_star = cs.fmin(cs.fmax(t_hat,0.0),1.0) # limit t
    temp_vec = s1 + t_star*s2s1 - p # vector pointing to closest point
    distance = cs.norm_2(temp_vec)
    return distance

def inside_ellipses(point: cs.SX, ellipse_param: List[cs.SX]):
    """Check if a point is inside a set of ellipses.
    
    Arguments:
        point: The (1*n)-dim target point.
        ellipse_param: Shape (5 or 6 * m) with m ellipses. 
                       Each ellipse is defined by (cx, cy, rx, ry, angle, alpha).
                       
    Returns:
        is_inside: The (1*m)-dim indicator vector. If inside, return positive value, else return negative value.
    """
    x, y = point[0], point[1]
    cx, cy, rx, ry, ang = ellipse_param[0], ellipse_param[1], ellipse_param[2], ellipse_param[3], ellipse_param[4]
    is_inside = 1 - ((x-cx)*cs.cos(ang)+(y-cy)*cs.sin(ang))**2 / (rx+1e-6)**2 - ((x-cx)*cs.sin(ang)-(y-cy)*cs.cos(ang))**2 / (ry+1e-6)**2
    return is_inside

def inside_cvx_polygon(point: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX):
    """Check if a point is inside a convex polygon defined by half-spaces.
    
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
    eq_mtx: cs.SX = cs.vertcat(b, -a0, -a1)
    result: cs.SX = cs.mtimes(eq_mtx.T, cs.vertcat(1, point[0], point[1]))
    is_inside = 1
    for i in range(result.shape[0]):
        is_inside *= cs.fmax(0, result[i])
    return is_inside

def outside_cvx_polygon(point: cs.SX, b: cs.SX, a0: cs.SX, a1: cs.SX):
    """Check if a point is outside a convex polygon defined by half-spaces.

    Arguments:
        point: The (1*n)-dim target point.
        b: Shape  (1*m) with m half-space offsets.
        a0: Shape (1*m) with m half-space weight vectors.
        a1: Shape (1*m) with m half-space weight vectors.

    Returns:
        is_outside: The (1*1)-dim indicator. If outside, return positive value, else return 0.

    Comments:
        Each half-space if defined as `b - [a0,a1]*[x,y]' > 0`.
        If sum(|min(0,all)|)>0, then the point is outside; Otherwise not.
    """
    eq_mtx: cs.SX = cs.vertcat(b, -a0, -a1)
    result: cs.SX = cs.mtimes(eq_mtx.T, cs.vertcat(1, point[0], point[1]))
    is_outside = 0
    for i in range(result.shape[0]):
        is_outside += cs.fmin(0, result[i]) ** 2
    return is_outside

def angle_between_vectors(l1: cs.SX, l2: cs.SX, degrees:bool=False):
    """Calculate the angle (radian) between two vectors.
    Each vector is defined by two points.
    """
    vec1 = l1[1,:] - l1[0,:]
    vec2 = l2[1,:] - l2[0,:]
    cos_angle = cs.dot(vec1, vec2) / (cs.norm_2(vec1)*cs.norm_2(vec2) + 1e-6)
    angle = cs.acos(cos_angle) * cs.sign(vec2[0]*vec1[1]-vec2[1]*vec1[0])
    if degrees:
        angle *= 180 / cs.pi
    return angle
