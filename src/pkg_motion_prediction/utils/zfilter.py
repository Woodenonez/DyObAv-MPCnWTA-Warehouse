import os, sys

import numpy as np

'''
File info:
    Name    - [zfilter]
    Author  - [Ze]
    Ref     - [https://en.wikipedia.org/wiki/Kalman_filter]
    Exe     - [Yes]
File description:
    (What does this file do?)
File content:
    KalmanFilter <class> - Build up a Kalman filter for prediction.
    model_CV     <func>  - A constant velocity motion model.
    fill_diag    <func>  - Create a np matrix with specific diagonal elements.
'''

class KalmanFilter():
    '''
    Description:
        Build up a Kalman filter for prediction.
    Arguments:
        state_space <list of matrices> - Contain the initial state and all matrices for a linear state space.
                                       - [X0, A, B, C D]
        P0          <np matrix>        - The initial state covairance matrix.
        Q           <np matrix>        - The state noise covairance matrix.
        R           <np matrix>        - The measurement noise covairance matrix.
    Attributes:
        tk <int>    - Discrete time step.
        Xs <ndarry> - States of every time step.
    Functions
        one_step     <run> - Evolve one step given control input U and measurement Y.
        append_state <set> - Append the given state to the state array Xs.
        predict_more <run> - Predict more steps to the future.
        predict      <run> - The predict step in KF.
        update       <run> - The update step in KF.
    '''
    def __init__(self, state_space, P0, Q, R, pred_offset=10):
        super().__init__()
        self.ss = state_space # [X0, A, B, C, D]
        self.X = state_space[0]
        self.P = P0
        self.Q = Q
        self.R = R
        self.tK = 0 # discrete time step
        self.offset = pred_offset

        self.Xs = state_space[0]

    def one_step(self, U, Y):
        self.tK += 1
        self.predict(U)
        self.update(U, Y)
        self.Xs = np.concatenate((self.Xs, self.X), axis=1)
        return self.X

    def append_state(self, X):
        self.Xs = np.concatenate((self.Xs, X), axis=1)
        self.tK += 1

    def predict_more(self, T, evolve_P=True):
        for _ in range(T):
            self.predict(np.zeros((np.shape(self.ss[2])[1],1)), evolve_P)
            self.Xs = np.concatenate((self.Xs, self.X), axis=1)

    def predict(self, U, evolve_P=True):
        A = self.ss[1]
        B = self.ss[2]
        self.X = np.dot(A, self.X) + np.dot(B, U)
        if evolve_P:
            self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q
        return self.X

    def update(self, U, Y):
        C = self.ss[3]
        D = self.ss[4]
        Yh = np.dot(C, self.X) + np.dot(D, U)
        S = self.R + np.dot(C, np.dot(self.P, C.T)) # innovation: covariance of Yh
        K = np.dot(self.P, np.dot(C.T, np.linalg.inv(S))) # Kalman gain
        self.X = self.X + np.dot(K, (Y-Yh))
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        return (self.X,K,S,Yh)

    def inference(self, traj):
        Y = [np.array(traj[1,:]), np.array(traj[2,:]), 
            np.array(traj[3,:]), np.array(traj[4,:])]
        for kf_i in range(len(Y) + self.offset):
            if kf_i<len(Y):
                self.one_step(np.array([[0]]), np.array(Y[kf_i]).reshape(2,1))
            else:
                self.predict(np.array([[0]]), evolve_P=False)
                self.append_state(self.X)
        return self.X, self.P

def model_CV(X0, Ts=1):
    A = np.array([[1,0,Ts,0], [0,1,0,Ts], [0,0,1,0], [0,0,0,1]])
    B = np.zeros((4,1))
    C = np.array([[1,0,0,0], [0,1,0,0]])
    D = np.zeros((2,1))
    return [X0, A, B, C, D]

def model_CA(X0, Ts=1):
    A = np.array([[1,0,Ts,0], [0,1,0,Ts], [0,0,1,0], [0,0,0,1]])
    B = np.array([[0,0], [0,0], [Ts,0], [0,Ts]])
    C = np.array([[1,0,0,0], [0,1,0,0]])
    D = np.zeros((2,2))
    return [X0, A, B, C, D]

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

