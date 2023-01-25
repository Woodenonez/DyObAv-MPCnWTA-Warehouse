import math

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as patches

N_HOR = 20
VEHICLE_WIDTH = 0.5
VEHICLE_MARGIN = 0.25

def plot_action(ax, action, ts, color='b'): # velocity or angular velocity
    time = np.linspace(0, ts*(len(action)), len(action))
    ax.plot(time, action, '-o', markersize = 4, linewidth=2, color=color)


def prepare_plot(fig, graph, start=None, end=None, legend_style='single', double_map=False, color_list=None, legend_list=None):
    """
    Prepare the plot.

    - graph: with .plot_map method to plot the map;
    - legend_style: 'single'(plot one object) or 'compare'(plot multiple objects);
    - double_map: if true, two identical maps are shown (so that one of them can be enlarged)
    """ 
    if not legend_style in ['single', 'compare']:
        raise ValueError(f'The legend style must be "single" or "compare", got {legend_style}.')
    if double_map:
        gs = GridSpec(6, 4, figure=fig)
    else:
        gs = GridSpec(3, 4, figure=fig)

    if double_map:
        vel_ax = fig.add_subplot(gs[0:2, :2])
        vel_ax.set_ylabel('Velocity [m/s]', fontsize=15)
        omega_ax = fig.add_subplot(gs[2:4, :2])
        omega_ax.set_ylabel('Angular velocity [rad/s]', fontsize=15)
        cost_ax = fig.add_subplot(gs[4:6, :2])
        cost_ax.set_xlabel('Time [s]', fontsize=15)
        cost_ax.set_ylabel('Cost', fontsize=15)
        path_ax = fig.add_subplot(gs[:3, 2:])
        path_ax.set_xlabel('X [m]', fontsize=15)
        path_ax.set_ylabel('Y [m]', fontsize=15)
        graph.plot_map(path_ax)
        path_ax1 = fig.add_subplot(gs[3:, 2:]) # this is the "double" map
        path_ax1.set_xlabel('Zoom-in', fontsize=15)
        path_ax1.set_ylabel('Zoom-in', fontsize=15)
        graph.plot_map(path_ax1)
    else:
        vel_ax = fig.add_subplot(gs[0, :2])
        vel_ax.set_ylabel('Velocity [m/s]', fontsize=15)
        omega_ax = fig.add_subplot(gs[1, :2])
        omega_ax.set_ylabel('Angular velocity [rad/s]', fontsize=15)
        cost_ax = fig.add_subplot(gs[2, :2])
        cost_ax.set_xlabel('Time [s]', fontsize=15)
        cost_ax.set_ylabel('Cost', fontsize=15)
        path_ax = fig.add_subplot(gs[:, 2:])
        path_ax.set_xlabel('X [m]', fontsize=15)
        path_ax.set_ylabel('Y [m]', fontsize=15)
        graph.plot_map(path_ax)
        path_ax1 = None

    if legend_style == 'single':
        legend_elems = [Line2D([0], [0], color='k', label='Original Boundary'),
                        Line2D([0], [0], color='g', label='Padded Boundary'),
                        Line2D([0], [0], color='r', label='Original Obstacles'),
                        Line2D([0], [0], color='y', label='Padded Obstacles'),
                        Line2D([0], [0], marker='o', color='b', label='Generated Path', alpha=0.5),
                        Line2D([0], [0], linewidth=0, marker='*', color='g', label='Start Position', alpha=0.5),
                        Line2D([0], [0], linewidth=0, marker='*', color='r', label='End Position'),
                        ]
    else:
        legend_elems = [Line2D([0], [0], linewidth=0, marker='*', color='g', label='Start Position', alpha=0.5),
                        Line2D([0], [0], linewidth=0, marker='*', color='r', label='End Position'),
                        ]
        for c, l in zip(color_list, legend_list):
            legend_elems.append(Line2D([0], [0], marker='o', color=c, label=l, alpha=0.5),)
    path_ax.legend(handles=legend_elems, fontsize=15) #, loc='lower left')
    path_ax.axis('equal')

    if start is not None:
        path_ax.plot(start[0], start[1], marker='*', color='g', markersize=15)
    if end is not None:
        path_ax.plot(end[0], end[1], marker='*', color='r', markersize=15)

    return vel_ax, omega_ax, cost_ax, path_ax, path_ax1

def update_plot(axes, ts, xx, xy, vel, omega, cost, color):
    vel_ax, omega_ax, cost_ax, path_ax, path_ax1 = axes
    plot_action(vel_ax, vel, ts, color)
    plot_action(omega_ax, omega, ts, color)
    plot_action(cost_ax, cost, ts, color)
    path_ax.plot(xx, xy, c=color, marker='o', alpha=0.5)
    if path_ax1 is not None:
        path_ax1.plot(xx, xy, c=color, marker='o', alpha=0.5)


def plot_results(graph, ts, x_coords, y_coords, vel, omega, cost, start, end, animation=False, scanner=None, video=False):
    if animation & (scanner is not None):
        plot_dynamic_results(graph, ts, x_coords, y_coords, vel, omega, cost, start, end, scanner, video)
    else:
        plot_static_results(graph, ts, x_coords, y_coords, vel, omega, cost, start, end)

def plot_static_results(graph, ts, xx, xy, vel, omega, cost, start=None, end=None):
    fig = plt.figure(constrained_layout=True)
    vel_ax, omega_ax, cost_ax, path_ax, _ = prepare_plot(fig, graph)
    plot_action(vel_ax, vel, ts)
    plot_action(omega_ax, omega, ts)
    plot_action(cost_ax, cost, ts)
    path_ax.plot(xx, xy, c='b', label='Path', marker='o', alpha=0.5)

def plot_dynamic_results(graph, ts, xx, xy, vel, omega, cost, start, end, scanner, make_video):
    if make_video:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        fig = plt.figure(figsize=(16,9))
    else:
        fig = plt.figure(constrained_layout=True)
    
    vel_ax, omega_ax, cost_ax, path_ax = prepare_plot(fig, graph)

    vel_line, = vel_ax.plot([1], '-o', markersize=4, linewidth=2)
    vel_ax.set_xlim(0, ts * len(xx))
    vel_ax.set_ylim(min(vel) - 0.1, max(vel) + 0.1)
    vel_ax.grid('on')

    omega_line, = omega_ax.plot([1], '-o', markersize=4, linewidth=2)
    omega_ax.set_xlim(0, ts * len(xx))
    omega_ax.set_ylim(min(omega) - 0.1, max(omega) + 0.1)
    omega_ax.grid('on')

    cost_line, = cost_ax.plot([1], '-o', markersize=4, linewidth=2)    
    cost_ax.set_xlim(0, ts * len(xx))
    cost_ax.set_ylim(min(cost) - 0.1, max(cost) + 0.1)

    path_ax.arrow(start[0], start[1], math.cos(start[2]), math.sin(start[2]), head_width=0.05, head_length=0.1, fc='k', ec='k')
    path_line, = path_ax.plot([1], '-ob', alpha=0.7, markersize=5)
    if make_video:
        fig.tight_layout()

    obs        = [object] * scanner.num_obstacles # NOTE: dynamic obstacles
    obs_padded = [object] * scanner.num_obstacles # NOTE: dynamic obstacles
    start_idx = 0
    for i in range(start_idx, len(xx)):
        time = np.linspace(0, ts*i, i)
        omega_line.set_data(time, omega[:i])
        vel_line.set_data(time, vel[:i])
        try:
            cost_line.set_data(time, cost[:i])
        except:
            cost_line.set_data(time, cost)
        path_line.set_data(xx[:i], xy[:i])

        veh = plt.Circle((xx[i], xy[i]), VEHICLE_WIDTH/2, color='b', alpha=0.7, label='Robot')
        path_ax.add_artist(veh)

        ### Plot obstacles # NOTE
        for idx in range(scanner.num_obstacles): # NOTE: Maybe different if the obstacle is different
            pos = scanner.get_obstacle_info(idx, i*ts, 'pos')
            x_radius, y_radius = scanner.get_obstacle_info(idx, i*ts, 'radius')
            angle = scanner.get_obstacle_info(idx, i*ts, 'angle')

            obs[idx] = patches.Ellipse(pos, x_radius*2, y_radius*2, angle/(2*math.pi)*360, color='r', label='Obstacle')
            x_rad_pad = x_radius + VEHICLE_WIDTH/2 + VEHICLE_MARGIN
            y_rad_pad = y_radius + VEHICLE_WIDTH/2 + VEHICLE_MARGIN
            obs_padded[idx] = patches.Ellipse(pos, x_rad_pad*2, y_rad_pad*2, angle/(2*math.pi)*360, color='y', alpha=0.7, label='Padded obstacle')
            
            path_ax.add_artist(obs_padded[idx])
            path_ax.add_artist(obs[idx])

        ## Plot predictions # NOTE
        pred = []
        for j, obstacle in enumerate(scanner.get_full_obstacle_list(i*ts, N_HOR, ts=ts)):
            for al, obsya in enumerate(obstacle):
                x,y,rx,ry,angle,_ = obsya
                pos = (x,y)
                this_ellipse = patches.Ellipse(pos, rx*2, ry*2, angle/(2*math.pi)*360, color='r', alpha=max(8-al,1)/20, label='Obstacle')
                pred.append(this_ellipse)
                path_ax.add_patch(this_ellipse)

        if make_video:
            canvas = FigureCanvas(fig) # put pixel buffer in numpy array
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            if i == start_idx:
                video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (mat.shape[1],mat.shape[0]))
            video.write(mat)
            print(f'\r Wrote frame {i+1}/{len(xx)}    ', end='')
        else:
            plt.draw()
            plt.pause(ts / 10)

            # while not plt.waitforbuttonpress():  # XXX press a button to continue
            #     pass
        
        veh.remove()
        for j in range(scanner.num_obstacles): # NOTE: dynamic obstacles
            obs[j].remove()
            obs_padded[j].remove()
        for j in range(len(pred)): # NOTE: dynamic obstacles (predictions)
            pred[j].remove()

    if make_video:
        video.release()
        cv2.destroyAllWindows()
        
    plt.show()
