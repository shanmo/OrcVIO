import numpy as np
import os, sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) +
    "/../third_party/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation")

from trajectory import Trajectory

class TrajEval():
    """
    this class evaluates the trajectory wrt groundtruth
    """
    def __init__(self):

        # ref http://www.cvlibs.net/datasets/kitti/eval_odometry.php
        self.subtraj_lengths = [x * 100 for x in range(1, 9)]

        self.result_dir = os.path.abspath(__file__ + "/../../../") + "/cache/"
        self.dir_path = os.path.abspath(__file__ + "/../../../") + "/python_scripts/" 

        # for debugging 
        # print(self.result_dir)
        # print(self.dir_path)

        trajectory_gt_filename = 'stamped_groundtruth.txt'
        self.trajectory_gt_filename = self.result_dir + trajectory_gt_filename

        trajectory_est_filename = 'stamped_traj_estimate.txt'
        self.trajectory_est_filename = self.result_dir + trajectory_est_filename

    def evaluate_and_plot(self):
        """
        evaluate the poses and show the plot
        """

        # <result_folder> should contain the groundtruth, trajectory estimate and
        # optionally the evaluation configuration as mentioned above.

        # remove old results
        folder_name = self.result_dir + 'plots/'
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)

        folder_name = self.result_dir + 'saved_results/'
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)

        evo_cmd = 'python ' + self.dir_path + 'third_party/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py ' \
                  + self.result_dir

        os.system(evo_cmd)

    def compute_kitti_relative_error(self):
        """
        Compute absolute error and relative error as defined in kitti
        """

        # static method to remove the cached error from a result folder
        Trajectory.remove_cached_error(self.result_dir, est_type='traj_est')

        # init with the result folder.
        # You can also specify the subtrajecotry lengths and alignment parameters in the initialization.
        traj = Trajectory(self.result_dir)

        # compute the absolute error
        traj.compute_absolute_error()

        # compute the relative errors at `subtraj_lengths`.
        traj.compute_relative_errors(self.subtraj_lengths)

        # compute the relative error at sub-trajectory lengths computed from the whole trajectory length.
        # traj.compute_relative_errors()

        # save the relative error to `cached_rel_err.pickle`
        # traj.cache_current_error()

        # write the error statistics to yaml files
        traj.write_kitti_errors_to_yaml()

    def evaluate(self):   

        self.evaluate_and_plot()
        self.compute_kitti_relative_error()

if __name__ == "__main__":

    import numpy as np
    import os, sys
    import shutil

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) +
        "/third_party/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation")

    from trajectory import Trajectory

    TE = TrajEval()
    TE.evaluate()
