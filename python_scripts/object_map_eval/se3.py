import numpy as np
import transforms3d as t3d
from scipy.linalg import expm, sinm, cosm

class SE3(object):
    """
    3d rigid transform.
    """

    def __init__(self, R, t):

        self.R = R
        self.t = t

    def matrix(self):

        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t

        return m

    def __mul__(self, T):

        R = self.R @ T.R
        t = self.R @ T.t + self.t

        return SE3(R, t)

    def inverse(self):

        return SE3(self.R.T, -self.R.T @ self.t)

def skew(a):
    """
    converts vector to skew symmetric matrix in batch
    :param a: size n x 3, input vector
    :return: S: size n x 3 x 3, skew symmetric matrix
    """

    S = np.empty(a.shape[:-1] + (3, 3))
    S[..., 0, 0].fill(0)
    S[..., 0, 1] = -a[..., 2]
    S[..., 0, 2] = a[..., 1]
    S[..., 1, 0] = a[..., 2]
    S[..., 1, 1].fill(0)
    S[..., 1, 2] = -a[..., 0]
    S[..., 2, 0] = -a[..., 1]
    S[..., 2, 1] = a[..., 0]
    S[..., 2, 2].fill(0)

    return S

def Hl_operator(omega):
    """
    implements Hl operator in eq 20 
    """

    omega_norm = np.linalg.norm(omega) 

    term1 = (1/2)*np.eye(3)
    term2 = np.nan_to_num((omega_norm - np.sin(omega_norm)) / (omega_norm**3)) * skew(omega)
    term3 = np.nan_to_num((2*(np.cos(omega_norm) - 1) + omega_norm**2) / (2*(omega_norm**4))) * (skew(omega) @ skew(omega))

    Hl = term1 + term2 + term3 
    
    return Hl

def Jl_operator(omega):
    """
    implements Jl operator in eq 20 
    """

    omega_norm = np.linalg.norm(omega) 

    term1 = np.eye(3)
    term2 = np.nan_to_num((1 - np.cos(omega_norm)) / (omega_norm**2)) * skew(omega)
    term3 = np.nan_to_num((omega_norm - np.sin(omega_norm)) / (omega_norm**3)) * (skew(omega) @ skew(omega))

    Jl = term1 + term2 + term3

    return Jl

def imu_kinametics_local(p, R, v, omega, acc, dt):
    """
    update SE3 pose based on imu measurements based on closed form integration 
    :param p: size 1x3, position before update
    :param R: size 3x3, rotation before update
    :param v: size 1x3, velocity before update
    :param omega: size 1x3, unbiased angular velocity
    :param acc: size 1x3, unbiased linear acceleration 
    :param dt: scalar of time difference
    :return: p_new: size 1x3, updated position
    R_new: size 3x3, update rotation
    """

    # update rotation
    delta_R = axangle2rot(dt*omega)
    R_new = R @ delta_R

    # Gravity vector in the world frame
    g = np.array([0., 0., -9.81])

    # update position 
    Hl = Hl_operator(dt*omega)
    p_new = p + dt*v + g*((dt**2)/2) + R @ Hl @ acc * (dt**2)

    # update velocity 
    Jl = Jl_operator(dt*omega)
    v_new = v + g*dt + R @ Jl @ acc * dt

    return R_new, p_new, v_new

def imu_kinametics(p, R, v, omega, acc, dt):
    """
    function to call 
    closed form propagation using right perturbation 
    original msckf propagation 
    """

    # closed form propagation using local frame kinematics
    R_new, p_new, v_new = imu_kinametics_local(p, R, v, omega, acc, dt)

    return R_new, p_new, v_new

def pose_kinametics(T, x):
    """
    update SE3 pose based on se3 element x
    :param T: size nx4x4, SE3 pose
    :param x: size nx1x6, se3 element
    :return: size nx4x4, update pose
    """

    return T @ axangle2pose(x)

def axangle2pose(x):
    """
    converts se3 element to SE3 in batch
    :param x: size n x 6, n se3 elements
    :return: size n x 4 x 4, n elements of SE(3)
    """

    return twist2pose(axangle2twist(x))

def twist2pose(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 4 x 4 pose (SE3) matrix 
  ''' 
  rotang = np.sqrt(np.sum(T[...,[2,0,1],[1,2,0]]**2,axis=-1)[...,None,None]) # n x 1
  Tn = np.nan_to_num(T / rotang)
  Tn2 = Tn@Tn
  Tn3 = Tn@Tn2
  eye = np.zeros_like(T)
  eye[...,[0,1,2,3],[0,1,2,3]] = 1.0
  return eye + T + (1.0 - np.cos(rotang))*Tn2 + (rotang - np.sin(rotang))*Tn3

def axangle2twist(x):
    """
    converts 6-vector to 4x4 hat form in se(3) in batch
    :param x: size n x 6, n se3 elements
    :return: size n x 4 x 4, n elements of se(3)
    """

    T = np.zeros(x.shape[:-1] + (4, 4))
    T[..., 0, 1] = -x[..., 5]
    T[..., 0, 2] = x[..., 4]
    T[..., 0, 3] = x[..., 0]
    T[..., 1, 0] = x[..., 5]
    T[..., 1, 2] = -x[..., 3]
    T[..., 1, 3] = x[..., 1]
    T[..., 2, 0] = -x[..., 4]
    T[..., 2, 1] = x[..., 3]
    T[..., 2, 3] = x[..., 2]

    return T

def inversePose(T):
    """
    performs batch inverse of transform matrix
    :param T: size n x 4 x 4, n elements of SE(3)
    :return: size n x 4 x 4, inverse of T
    """

    iT = np.empty_like(T)
    iT[..., 0, 0], iT[..., 0, 1], iT[..., 0, 2] = T[..., 0, 0], T[..., 1, 0], T[..., 2, 0]
    iT[..., 1, 0], iT[..., 1, 1], iT[..., 1, 2] = T[..., 0, 1], T[..., 1, 1], T[..., 2, 1]
    iT[..., 2, 0], iT[..., 2, 1], iT[..., 2, 2] = T[..., 0, 2], T[..., 1, 2], T[..., 2, 2]
    iT[..., :3, 3] = -np.squeeze(iT[..., :3, :3] @ T[..., :3, 3, None])
    iT[..., 3, :] = T[..., 3, :]

    return iT

def axangle2rot(a):
    '''
    converts axis angle to SO3 in batch
    @Input:
      a = n x 3 = n axis-angle elements
    @Output:
      R = n x 3 x 3 = n elements of SO(3)
    '''

    na = np.linalg.norm(a, axis=-1)  # n x 1
    # cannot add epsilon to denominator
    ana = np.nan_to_num(a / na[..., None])  # n x 3
    ca, sa = np.cos(na), np.sin(na)  # n x 1
    mc_ana = ana * (1 - ca[..., None])  # n x 3
    sa_ana = ana * sa[..., None]  # n x 3

    R = np.empty(a.shape + (3,))
    R[..., 0, 0] = mc_ana[..., 0] * ana[..., 0] + ca
    R[..., 0, 1] = mc_ana[..., 0] * ana[..., 1] - sa_ana[..., 2]
    R[..., 0, 2] = mc_ana[..., 0] * ana[..., 2] + sa_ana[..., 1]
    R[..., 1, 0] = mc_ana[..., 0] * ana[..., 1] + sa_ana[..., 2]
    R[..., 1, 1] = mc_ana[..., 1] * ana[..., 1] + ca
    R[..., 1, 2] = mc_ana[..., 2] * ana[..., 1] - sa_ana[..., 0]
    R[..., 2, 0] = mc_ana[..., 0] * ana[..., 2] - sa_ana[..., 1]
    R[..., 2, 1] = mc_ana[..., 1] * ana[..., 2] + sa_ana[..., 0]
    R[..., 2, 2] = mc_ana[..., 2] * ana[..., 2] + ca

    return R

def odotOperator(ph):
    '''
    @Input:
      ph = n x 4 = points in homogeneous coordinates
    @Output:
    odot(ph) = n x 4 x 6
    '''

    zz = np.zeros(ph.shape + (6,))
    zz[...,:3,3:6] = -skew(ph[...,:3])
    zz[...,0,0],zz[...,1,1],zz[...,2,2] = ph[...,3],ph[...,3],ph[...,3]

    return zz

def circledCirc(ph):
    '''
    @Input:
      ph = n x 4 = points in homogeneous coordinates
    @Output:
    circledCirc(ph) = n x 6 x 4
    '''

    zz = np.zeros(ph.shape[:-1] + (6,4))
    zz[...,3:,:3] = -skew(ph[...,:3])
    zz[...,:3,3] = ph[...,:3]

    return zz

def poseSE32SE2(T, force_z_to_zero_flag = False):

    yaw = t3d.euler.mat2euler(T[:3, :3], axes='rzyx')[0]

    if not force_z_to_zero_flag:

        # note that we keep T[2, 3] instead of force z = 0
        T = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, T[0, 3]],
                      [np.sin(yaw), np.cos(yaw), 0.0, T[1, 3]],
                      [0.0, 0.0, 1.0, T[2, 3]],
                      [0.0, 0.0, 0.0, 1.0]])

    else:

        T = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, T[0, 3]],
                      [np.sin(yaw), np.cos(yaw), 0.0, T[1, 3]],
                      [0.0, 0.0, 1.0, 0],
                      [0.0, 0.0, 0.0, 1.0]])

    return T

def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0,0] - R[1,1] - R[2,2]
            q = [t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]]
        else:
            t = 1 - R[0,0] + R[1,1] - R[2,2]
            q = [R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2], R[2, 0]-R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0,0] - R[1,1] + R[2,2]
            q = [R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t, R[0, 1]-R[1, 0]]
        else:
            t = 1 + R[0,0] + R[1,1] + R[2,2]
            q = [R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t]

    q = np.array(q) # * 0.5 / np.sqrt(t)
    return q / np.linalg.norm(q)

def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    q = np.squeeze(q)

    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
    return R

def perturb_T(T):
    """
    this function perturbs T if it's too close to the origin 
    T size is 4 x 4 
    """
    if np.allclose(T[:3, 3], 0):
        warnings.warn("Numerical derivatives around origin are bad, because"
                      + " the distance function is non-differentiable. Choosing a random"
                      + " position. ")
        T[:3, 3] = np.random.rand(3) * 2 - 1
    return T 