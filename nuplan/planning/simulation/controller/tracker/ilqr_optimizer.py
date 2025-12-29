import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import ILQRSolver

DoubleMatrix = npt.NDArray[np.float64]


class ILQROptimizer:
    """
    Tracker using an iLQR solver with a kinematic bicycle model.
    """

    def __init__(self, n_horizon: int, ilqr_solver: ILQRSolver) -> None:
        """
        Initialize tracker parameters, primarily the iLQR solver.
        :param n_horizon: Maximum time horizon (number of discrete time steps) that we should plan ahead.
                          Please note the associated discretization_time is specified in the ilqr_solver.
        :param ilqr_solver: Solver used to compute inputs to apply.
        """
        assert n_horizon > 0, "The time horizon length should be positive."
        self._n_horizon = n_horizon

        self._ilqr_solver = ilqr_solver

    def optimize_trajectory(
        self,
        initial_state: EgoState,
        reference_trajectory: DoubleMatrix,
    ) -> DoubleMatrix:
        """Inherited, see superclass."""
        current_state: DoubleMatrix = np.array(
            [
                initial_state.rear_axle.x,
                initial_state.rear_axle.y,
                initial_state.rear_axle.heading,
                initial_state.dynamic_car_state.rear_axle_velocity_2d.x,
                initial_state.tire_steering_angle,
            ]
        )

        # Run the iLQR solver to get the optimal input sequence to track the reference trajectory.
        solutions = self._ilqr_solver.solve(current_state, reference_trajectory)
        optimal_trajectory = solutions[-1].state_trajectory


        return optimal_trajectory