from main_base import MainBase

def colored_print(r, g, b, text):
    print("\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)) 

def main(tracker_type, predictor_type, max_run_time_step=120):
    """
    - Tracker type: 'mpc' or 'dwa'.
    - Predictor type: 'kfmp' or 'cvmp' or 'mmp' or `None`.
    """
    colored_print(0, 255, 0, 'Tracker type: %s, Predictor type: %s' % (tracker_type, predictor_type))
    main_pro = MainBase(max_run_time_step=max_run_time_step, cfg_fname='global_setting_warehouse.yaml', evaluation=False)
    main_pro.run(tracker_type=tracker_type, predictor_type=predictor_type, plot_graph=False)


if __name__ == '__main__':
    main(tracker_type='mpc', predictor_type='mmp')
    # main(tracker_type='mpc', predictor_type='kfmp') # XXX
    # main(tracker_type='mpc', predictor_type='cvmp')
    # main(tracker_type='dwa', predictor_type=None)
    # main(tracker_type='dwa', predictor_type='cvmp')
