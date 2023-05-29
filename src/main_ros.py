#!/usr/bit/env python3

from re import I
import rospy
import message_filters


from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistWithCovariance
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as Plan
from geometry_msgs.msg import PoseArray

from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32
from nav_msgs.srv import GetPlan

from threading import Thread
import copy
import numpy
import math

from tf import transformations

from util import mapnet
from util import basic_agent


class UnicyclePose:

    def __init__(self, x = 0.0, y = 0.0, theta = 0.0, time_stamp = rospy.Time(0), frame_id = 'map'):
        self.x = x
        self.y = y
        self.theta = theta
        self.time_stamp = time_stamp
        self.frame_id = frame_id

    def __str__(self) -> str:
        result = f"x: {self.x:0.3f}, y:{self.y:0.3f}, theta :{self.theta:0.3f}, TimeStamp: {self.time_stamp}, FrameID: {self.frame_id} "
        return result


def rosPose2D(x: float, y: float, theta: float)->Pose:
    result = Pose()
    result.position.x = x
    result.position.y = y
    q = transformations.quaternion_from_euler(0.0, 0.0, theta)
    result.orientation.x = q[0]
    result.orientation.y = q[1]
    result.orientation.z = q[2]
    result.orientation.w = q[3]
    return result

def rosPoseStamped2D(pose: Pose, frame_id: str, seq = 0)->PoseStamped:
    result = PoseStamped()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    header.seq = seq
    result.header = header
    result.pose = pose
    return result

def rosPoseWithCovariance2D(x: float, y: float, theta: float, covariance = [0.0] * 36)->PoseWithCovariance:
    result = PoseWithCovariance()
    result.pose = rosPose2D(x, y, theta)
    result.covariance = covariance
    return result

def rosPoseWithCovarianceStamp2D(x: float, y: float, theta: float, covariance = [0.0] * 36, frame_id = 'world', seq = 0)->PoseWithCovarianceStamped:
    result = PoseWithCovarianceStamped()
    result.pose.pose = rosPose2D(x, y, theta)
    result.pose.covariance = covariance
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    header.seq = seq
    result.header = header
    return result

def rosTwist2D(v: float, omega: float)->Twist:
    result = Twist()
    result.linear.x = v
    result.angular.z = omega
    return result

def rosTwistWithCovariance2D(v: float, omega: float, covariance = [0.0] * 36)->TwistWithCovariance:
    result = TwistWithCovariance()
    result.twist = rosTwist2D(v, omega)
    result.covariance = covariance
    return result

def rosOdometry2D(pose: PoseWithCovariance, twist: TwistWithCovariance, frame_id = 'world', child_frame_id = 'odom', seq = 0)->Odometry:
    result = Odometry()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    header.seq = seq
    result.header = header
    result.child_frame_id = child_frame_id
    result.pose = pose
    result.twist = twist
    return result

def rosPath2D(path:list, frame_id = 'map')->Plan:
    result = Plan()
    seq = 0
    for item in path:
        p = rosPoseStamped2D( pose = rosPose2D(x = item[0], y = item[1], theta = item[2]), frame_id = frame_id, seq = seq)
        result.poses.append(p)
        seq = seq + 1
    result.header.frame_id = frame_id
    result.header.stamp = rospy.Time.now()
    return result

def cleanUp():
    print("shutdown time!")

def getGlobalPlan(start: PoseStamped, end: PoseStamped)->Plan:
    rospy.wait_for_service('/global_path_planner/make_plan')
    try:
        planner_server = rospy.ServiceProxy('/global_path_planner/make_plan', GetPlan)
        plan = planner_server(start, end, 0.1)
        return plan.plan
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def publishPathPlan(path: Plan, pub_rate = 10, topic_name = '/global_plan'):
    rate = rospy.Rate(pub_rate)
    pub = rospy.Publisher(topic_name, Plan, queue_size = 10)
    while not rospy.is_shutdown():
        pub.publish(path)
        rate.sleep()

def poseStampedtoUnicyclePose(pose_stamped: PoseStamped) -> UnicyclePose:
    x = pose_stamped.pose.position.x
    y = pose_stamped.pose.position.y
    quaternion = (  pose_stamped.pose.orientation.x,
                    pose_stamped.pose.orientation.y,
                    pose_stamped.pose.orientation.z,
                    pose_stamped.pose.orientation.w
                )
    eulerRPY = transformations.euler_from_quaternion(quaternion)
    theta = eulerRPY[-1]
    result = UnicyclePose(x = x, y = y, theta = theta, time_stamp = pose_stamped.header.stamp, frame_id = pose_stamped.header.frame_id)
    return result

def poseWithCovarianceStamptoUnicyclePose(pose_stamped: PoseWithCovarianceStamped) -> UnicyclePose:
    x = pose_stamped.pose.pose.position.x
    y = pose_stamped.pose.pose.position.y
    quaternion = (  pose_stamped.pose.pose.orientation.x,
                    pose_stamped.pose.pose.orientation.y,
                    pose_stamped.pose.pose.orientation.z,
                    pose_stamped.pose.pose.orientation.w
                )
    eulerRPY = transformations.euler_from_quaternion(quaternion)
    theta = eulerRPY[-1]
    result = UnicyclePose(x = x, y = y, theta = theta, time_stamp = pose_stamped.header.stamp, frame_id = pose_stamped.header.frame_id)
    return result

def odometrytoUnicyclePose(odom: Odometry) -> UnicyclePose:
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    quaternion = (  odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w
                )
    eulerRPY = transformations.euler_from_quaternion(quaternion)
    theta = eulerRPY[-1]
    result = UnicyclePose(x = x, y = y, theta = theta, time_stamp = odom.header.stamp, frame_id = odom.header.frame_id)
    return result

def setRobotVelocity(v: float, omega: float):
    
    velocity_command = Twist()
    
    velocity_command.linear.x = v
    velocity_command.linear.y = 0
    velocity_command.linear.z = 0

    velocity_command.angular.x = 0
    velocity_command.angular.y = 0
    velocity_command.angular.z = omega

    velocity_publisher = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 1)
    velocity_publisher.publish(velocity_command)

def getRobotOdometry(estimated = False, timeout = 1) -> UnicyclePose:
    result = None
    if estimated:
        robot_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout = timeout)
        result = poseWithCovarianceStamptoUnicyclePose(robot_pose)
    else:
        robot_pose = rospy.wait_for_message('/base_pose_ground_truth', Odometry, timeout = timeout)
        result = odometrytoUnicyclePose(robot_pose)
    return result


actors_traj_history = []
def getActorsPoses(timeout = 1.0) -> list:
    result = []
    # HH_OF
    actor1_pose = rospy.wait_for_message('/actor1_pose', Odometry, timeout = timeout/2.0)
    result.append(odometrytoUnicyclePose(actor1_pose))
    p = result[-1]
    actors_traj_history.append(numpy.array([p.x, p.y]))
    return result


if __name__ == "__main__":
    import pathlib
    import numpy as np

    from util.basic_objclass import *

    from util import mapnet # process map
    from main_pre import prepare_map, prepare_params
    from mpc_planner.mpc_interface import MpcInterface, CoordTransform
    from motion_prediction.mmp_interface import MmpInterface
    from motion_prediction.util import utils_test # for CGF

    publish_plan_thrd = None

    rospy.init_node('main')
    plan = []
    start = UnicyclePose(x = 0.0, y = 0.0, theta = 0.0)
    end = UnicyclePose(x = 8.0, y = 8.0, theta = 0.0)



    ### ZE CODE ###
    # base_path = Path(g_plan)
    
    ### Global custom
    CONFIG_FILE = 'global_setting_warehouse.yml'

    ### Global load
    ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
    params = prepare_params(CONFIG_FILE, ROOT_DIR)
    SCENE   = params['SCENE']
    MMP_CFG = params['MMP_CFG']
    MPC_CFG = params['MPC_CFG']
    STIME   = params['STIME']
    PRED_OFFSET = params['PRED_OFFSET']

    SCALE2NN    = params['SCALE2NN']
    SCALE2REAL  = params['SCALE2REAL']
    IMAGE_AXIS  = params['IMAGE_AXIS']
    CORNER_COORDS = params['CORNER_COORDS']
    SIM_WIDTH  = params['SIM_WIDTH']
    SIM_HEIGHT = params['SIM_HEIGHT']

    ROBOT_SIZE = 0.5
    ROBOT_VMAX = 2.0

    ct2real = CoordTransform(scale=SCALE2REAL, offsetx_after=CORNER_COORDS[0], offsety_after=CORNER_COORDS[1], 
                             y_reverse=~IMAGE_AXIS, y_max_before=SIM_HEIGHT)

    the_map, ref_map = prepare_map(SCENE, ROOT_DIR, inversed_pixel=True)
    map_info = {'map_image':the_map, 'threshold':120}
    scene_graph = mapnet.SceneGraph(scene=SCENE, map_type='occupancy', map_info=map_info)
    geo_map_rescale = scene_graph.base_map.get_geometric_map()
    geo_map_rescale.coords_cvt(ct2real)
    geo_map_rescale.inflation(ROBOT_SIZE)
    # base_path_rescale = Path([Node(list(ct2real(np.array(x)))) for x in base_path()])


    the_planner = basic_agent.Planner(scene_graph.NG)
    _, backup_paths = the_planner.k_shortest_paths(source=8, target=14, k=1)
    base_path = backup_paths[0]
    # base_path_rescale = copy.deepcopy(base_path)
    # base_path_rescale.rescale(SCALE2REAL)
    # base_path_rescale = Path([Node(list(ct2real(np.array(x)))) for x in base_path()])
    # base_path_rescale = Path([Node(1.0,-4,1.57),Node(1.5,11,0),Node(8,11,0)])
    #HH_OF
    base_path_rescale = Path([Node(10.0,10.0,-1.57),Node(10,2.2,-1.57),Node(0,2.2,-3.14),Node(0,-8,-1.57)])   

    #current_state = getRobotPose() # NOTE from ROS
    robot_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout = 1)
    robot_odom = rospy.wait_for_message('/base_pose_ground_truth', Odometry, timeout = 1)
    
    robot_pose_stamped = PoseStamped()
    robot_pose_stamped.header = robot_pose.header
    robot_pose_stamped.pose = robot_pose.pose.pose
    current_state = poseStampedtoUnicyclePose(robot_pose_stamped)




    current_state = np.array([current_state.x, current_state.y, current_state.theta])
    motion_predictor = MmpInterface(MMP_CFG)
    traj_generator   = MpcInterface(MPC_CFG, current_state, geo_map_rescale, external_base_path=base_path_rescale, init_build=False)

    global_path = rosPath2D(path = traj_generator.ref_traj, frame_id = 'map')
    publish_plan_thrd = Thread(target = publishPathPlan, args = (global_path,))
    publish_plan_thrd.start()


    polygon_publishers = []
    for i in range(20):
        polygon_publishers.append(rospy.Publisher(f'/obst_{i}', PolygonStamped, queue_size=10)) 


    velocity_odom_publisher = rospy.Publisher('/cmd_vel_odom', Odometry, queue_size = 10)
    
    pose_estim_publisher = rospy.Publisher('/estimated_ped_pose', PoseArray, queue_size = 10)


    DONE = False
    PRED_OFFSET = 20
    while not DONE:

        res = getActorsPoses()
        past_traj = Trajectory([Node(list(ct2real(x,False))) for x in actors_traj_history])
        hypos_list = motion_predictor.get_motion_prediction(past_traj, ref_map, PRED_OFFSET, SCALE2NN, batch_size = 4)
        
        hypos_list = [ct2real.cvt_coords(x[:,0], x[:,1]) for x in hypos_list] # cvt2real
        # hypos_list = None
        if hypos_list is not None:
            ### CGF

            mu_list_list  = []
            std_list_list = []
            for i in range(PRED_OFFSET):
                hyposM = hypos_list[i]
                hypos_clusters    = utils_test.fit_DBSCAN(hyposM, eps=1, min_sample=2) # DBSCAN
                mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters, enlarge=2) # Gaussian fitting
                mu_list_list.append(mu_list)
                std_list_list.append(std_list)
            ### Get dynamic obstacle list for MPC
            n_obs = 0
            # HH_OF
            estimated_poses = PoseArray()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "map"
            estimated_poses.header = header


            for L in mu_list_list:
                for l in L:
                    estimated_poses.poses.append(rosPose2D(x = l[0], y = l[1], theta = 0.0))
            pose_estim_publisher.publish(estimated_poses)

            for mu_list in mu_list_list:
                if len(mu_list)>n_obs:
                    n_obs = len(mu_list)
            full_dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*traj_generator.config.N_hor for _ in range(n_obs)]
            for Tt, (mu_list, std_list) in enumerate(zip(mu_list_list, std_list_list)):
                for Nn, (mu, std) in enumerate(zip(mu_list, std_list)): # at each time offset
                    full_dyn_obs_list[Nn][Tt] = [mu[0], mu[1], std[0], std[1], 0, 1]
        else:
            full_dyn_obs_list = None
    
        # current_state = getRobotPose() # NOTE from ROS
        robot_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout = 1)
        robot_odom = rospy.wait_for_message('/base_pose_ground_truth', Odometry, timeout = 1)
        
        robot_pose_stamped = PoseStamped()
        robot_pose_stamped.header = robot_pose.header
        robot_pose_stamped.pose = robot_pose.pose.pose
        current_state = poseStampedtoUnicyclePose(robot_pose_stamped)


        current_state = np.array([current_state.x, current_state.y, current_state.theta])
        #print("current_state")
        #print("\ncurrent_state")
        #print(current_state)
        traj_generator.traj_gen.set_current_state(current_state) # NOTE: This is the correction of the state in trajectory generator!!!
        actions, pred_states, _, cost, the_obs_list = traj_generator.run_step('super', full_dyn_obs_list, True)
        action = actions[0]

        for i in range(len(the_obs_list)):
            polygon = PolygonStamped()
            polygon.header.stamp = rospy.Time.now()
            polygon.header.frame_id = 'map'
            for j in range(len(the_obs_list[i])):
                p = Point32()
                p.x = the_obs_list[i][j][0]
                p.y = the_obs_list[i][j][1]
                p.z = 0.0
                polygon.polygon.points.append(p)
            polygon_publishers[i].publish(polygon)

        #print(f"\nLinear Velocity: {action[0]} --- Angular Velocity: {action[1]}")
        setRobotVelocity(action[0], action[1])
        print(action[0])
        vel_command = Odometry()
        vel_command.header = robot_odom.header
        vel_command.child_frame_id = "odom"
        vel_command.pose = robot_pose.pose
        vel_command.twist = TwistWithCovariance()
        vel_command.twist.twist.linear.x = action[0]
        vel_command.twist.twist.angular.z = action[1]

        velocity_odom_publisher.publish(vel_command)


    ### ZE CODE ###


    publish_plan_thrd.join()
    rospy.on_shutdown(cleanUp)
