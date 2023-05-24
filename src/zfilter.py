import numpy as np
from typing import List


class KalmanFilter():
    """Build up a Kalman filter for prediction.

    Attributes:
        Xs: States of every time step.

    Functions
        predict: The predict step in KF.
        update: The update step in KF.
        one_step: Evolve one step given control input U and measurement Y.
        predict_more: Predict more steps to the future.
    """
    def __init__(self, state_space: List[np.ndarray], P0: np.ndarray, Q: np.ndarray, R: np.ndarray, pred_offset:int=10):
        """
        Arguments:
            state_space: Contain all matrices for a linear state space [A, B, C D].
            P0: The initial state covariance matrix.
            Q: The state noise covariance matrix.
            R: The measurement noise covariance matrix.
            pred_offset: The maximum number of steps to predict.
        """
        super().__init__()
        self.ss = state_space # [A, B, C, D]
        self.P = P0
        self.Q = Q
        self.R = R
        self.offset = pred_offset

        self.ns = self.ss[0].shape[0] # number of states
        self.nu = self.ss[1].shape[1] # number of inputs

    def set_init_state(self, init_state: np.ndarray):
        """Set the initial state (nx1)."""
        self.X = init_state
        self.Xs = init_state

    def set_state_space(self, state_space: List[np.ndarray]):
        """Set the state space."""
        self.ss = state_space

    def predict(self, U, evolve_P=True):
        A = self.ss[0]
        B = self.ss[1]
        self.X = np.dot(A, self.X) + np.dot(B, U)
        if evolve_P:
            self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q
        return self.X

    def update(self, U, Y):
        C = self.ss[2]
        D = self.ss[3]
        Yh = np.dot(C, self.X) + np.dot(D, U)
        S = self.R + np.dot(C, np.dot(self.P, C.T)) # innovation: covariance of Yh
        K = np.dot(self.P, np.dot(C.T, np.linalg.inv(S))) # Kalman gain
        self.X = self.X + np.dot(K, (Y-Yh))
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        return self.X, K, S, Yh

    def one_step(self, U, Y):
        self.predict(U)
        self.update(U, Y)
        self.Xs = np.concatenate((self.Xs, self.X), axis=1)
        return self.X

    def inference(self, traj: np.ndarray):
        """Inference the trajectory (mx2)."""
        traj_len = traj.shape[0]
        for kf_i in range(traj_len-1 + self.offset):
            if kf_i < traj_len-1:
                self.one_step(np.zeros((self.nu, 1)), traj[kf_i+1, :].reshape(2,1))
            else:
                self.predict(np.zeros((self.nu, 1)), evolve_P=False)
                self.Xs = np.concatenate((self.Xs, self.X), axis=1)
        return self.X, self.P

def model_CV(ts:float=1.0) -> List[np.ndarray]:
    """Return a linear state space model for a constant velocity model."""
    A = np.array([[1,0,ts,0], [0,1,0,ts], [0,0,1,0], [0,0,0,1]])
    B = np.zeros((4,1))
    C = np.array([[1,0,0,0], [0,1,0,0]])
    D = np.zeros((2,1))
    return [A, B, C, D]

def model_CA(ts:float=1.0) -> List[np.ndarray]:
    """Return a linear state space model for a constant acceleration model."""
    A = np.array([[1,0,ts,0], [0,1,0,ts], [0,0,1,0], [0,0,0,1]])
    B = np.array([[0,0], [0,0], [ts,0], [0,ts]])
    C = np.array([[1,0,0,0], [0,1,0,0]])
    D = np.zeros((2,2))
    return [A, B, C, D]

def model_CT(ts:float, state: np.ndarray, omega: float) -> List[np.ndarray]:
    """Return a linear state space model for a Coordinated Turn model assuming constant velocity.
    
    Arguments:
        state: [x, y, v, phi]
        omega: The turn rate.
    
    Comments:
        Assume the velocity is constant.
    """
    x, y, v, phi = state[0], state[1], state[2], state[3]
    A = np.array([
            [1, 0, ts * np.cos(phi), -v * ts * np.sin(phi)],
            [0, 1, ts * np.sin(phi),  v * ts * np.cos(phi)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    B = np.array([
        [-v * ts * np.sin(phi), v * (np.cos(phi) - np.cos(phi + omega * ts)) / omega],
        [ v * ts * np.cos(phi), v * (np.sin(phi) - np.sin(phi + omega * ts)) / omega],
        [0, 0],
        [0, ts]
    ])

    C = np.array([[1,0,0,0], [0,1,0,0]])
    D = np.zeros((2,2))
    return [A, B, C, D]

def fill_diag(diag):
    M = np.zeros((len(diag),len(diag)))
    for i in range(len(diag)):
        M[i,i] = diag[i]
    return M



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch import tensor as ts

    X0 = np.array([[0,0,0,0]]).transpose()
    model1 = model_CV(X0)
    model2 = model_CA(X0)

    P0 = fill_diag((1,1,1,1))
    Q1 = np.eye(4)#fill_diag((1,1,1,100))
    Q2 = np.eye(4)
    R  = np.eye(2)
    KF1 = KalmanFilter(model1,P0,Q1,R)
    KF2 = KalmanFilter(model2,P0,Q2,R)

    Y = [(1,0),(2.3,0),(2.5,1),(2.5,2),(2.5,3),(3,3),(4,3)]
    U1 = np.array([[0]])
    U2 = np.array([[0], [0]])

    fig, ax = plt.subplots()

    for i in range(len(Y)):
        U2 = (np.random.rand(2,1)-0.5) / 10
        if i<len(Y):
            KF1.one_step(U1, np.array(Y[i]).reshape(2,1))
            KF2.one_step(U2, np.array(Y[i]).reshape(2,1))
        else:
            KF1.predict(U1,evolve_P=True)
            KF2.predict(U2,evolve_P=True)
            KF1.append_state(KF1.X)
            KF2.append_state(KF2.X)
        patch1 = patches.Ellipse(KF1.X[:2].reshape(-1), KF1.P[0,0], KF1.P[1,1], fc='g')
        patch2 = patches.Ellipse(KF2.X[:2].reshape(-1), KF2.P[0,0], KF2.P[1,1], fc='y')
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        plt.plot(KF1.Xs[0,:], KF1.Xs[1,:], 'bo-')
        plt.plot(KF2.Xs[0,:], KF2.Xs[1,:], 'go-')
        plt.plot(np.array(Y)[:,0], np.array(Y)[:,1], 'rx')
        plt.pause(0.9)

    plt.show()

