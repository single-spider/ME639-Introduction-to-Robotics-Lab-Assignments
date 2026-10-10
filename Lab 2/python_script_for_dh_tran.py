import numpy as np


def dh_transform(a, alpha, d, theta):
    """
    Create a 4x4 homogeneous transformation matrix
    from DH parameters.

    Parameters:
        a      : link length
        alpha  : link twist in radians
        d      : link offset
        theta  : joint angle in radians

    Returns:
        4x4 numpy array
    """

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ], dtype=float)

    return T


def modified_dh_transform(a, alpha, d, theta):
    """Modified DH: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st, 0, a],
        [ca * st, ca * ct, -sa, -d * sa],
        [sa * st, sa * ct, ca, d * ca],
        [0, 0, 0, 1],
    ], dtype=float)


def forward_kinematics(dh_table, verbose=False, convention="standard"):
    """
    Compute forward kinematics from a DH table.

    Each row must contain:
        [a, alpha, d, theta]

    Returns:
        Final 4x4 transformation matrix
    """

    T_total = np.eye(4)

    for i, row in enumerate(dh_table):
        a, alpha, d, theta = row

        if convention == "standard":
            T_i = dh_transform(a, alpha, d, theta)
        elif convention == "modified":
            T_i = modified_dh_transform(a, alpha, d, theta)
        else:
            raise ValueError("convention must be 'standard' or 'modified'")

        if verbose:
            print(f"\nTransformation T_{i}_{i+1}:")
            print(np.round(T_i, 4))

        T_total = T_total @ T_i

        if verbose:
            print(f"\nCumulative transformation T_0_{i+1}:")
            print(np.round(T_total, 4))

    return T_total


def print_pose(T):
    """
    Print position and orientation from a homogeneous matrix.
    """

    position = T[:3, 3]
    rotation = T[:3, :3]

    print("\nFinal Homogeneous Transformation Matrix:")
    print(np.round(T, 4))

    print("\nEnd-Effector Position:")
    print(np.round(position, 4))

    print("\nEnd-Effector Rotation Matrix:")
    print(np.round(rotation, 4))


# ============================================================
# GENERIC 6-DOF ROBOT
# ============================================================

def generic_6dof():
    print("\n======================================")
    print("GENERIC 6-DOF ROBOT")
    print("======================================")

    # Joint angles in degrees
    joint_angles_deg = [
        0,
        0,
        0,
        0,
        0,
        0
    ]

    # Convert degrees -> radians
    q = np.deg2rad(joint_angles_deg)

    # --------------------------------------------------------
    # Example DH parameters
    #
    # Replace these values with the DH parameters
    # of whichever 6-DOF robot you are given.
    #
    # Format:
    # [a, alpha, d, theta]
    # --------------------------------------------------------

    dh_table = np.array([
        [0.0,     np.pi / 2,   0.4, q[0]],
        [0.3,     0.0,         0.0, q[1]],
        [0.25,    0.0,         0.0, q[2]],
        [0.0,     np.pi / 2,   0.2, q[3]],
        [0.0,    -np.pi / 2,   0.0, q[4]],
        [0.0,     0.0,         0.1, q[5]]
    ])

    T_ee = forward_kinematics(
        dh_table,
        verbose=True
    )

    print_pose(T_ee)


# ============================================================
# FRANKA RESEARCH 3 - 7 DOF
# ============================================================

def franka_7dof(joint_angles_deg=None, verbose=True):
    if verbose:
        print("\n======================================")
        print("FRANKA PANDA - 7 DOF")
        print("======================================")

    # --------------------------------------------------------
    # Joint angles
    #
    # Change these values to whatever configuration
    # you want to test.
    # --------------------------------------------------------

    if joint_angles_deg is None:
        joint_angles_deg = [0, 0, 0, -45, 0, 45, 0]
    if len(joint_angles_deg) != 7:
        raise ValueError("Franka needs seven joint angles")

    q = np.deg2rad(joint_angles_deg)

    # --------------------------------------------------------
    # Franka Research 3 modified DH parameters for the supplied Panda model.
    #
    # Format:
    # [a, alpha, d, theta]
    #
    # Units:
    # a, d     -> metres
    # alpha    -> radians
    # theta    -> radians
    # --------------------------------------------------------

    dh_franka = np.array([

        # Joint 1
        [
            0.0,
            0.0,
            0.333,
            q[0]
        ],

        # Joint 2
        [
            0.0,
            -np.pi / 2,
            0.0,
            q[1]
        ],

        # Joint 3
        [
            0.0,
            np.pi / 2,
            0.316,
            q[2]
        ],

        # Joint 4
        [
            0.0825,
            np.pi / 2,
            0.0,
            q[3]
        ],

        # Joint 5
        [
            -0.0825,
            -np.pi / 2,
            0.384,
            q[4]
        ],

        # Joint 6
        [
            0.0,
            np.pi / 2,
            0.0,
            q[5]
        ],

        # Joint 7
        [
            0.088,
            np.pi / 2,
            0.0,
            q[6]
        ]
    ])

    # Base -> Joint 7
    T_07 = forward_kinematics(
        dh_franka,
        verbose=verbose,
        convention="modified",
    )

    if verbose:
        print("\n======================================")
        print("BASE -> JOINT 7")
        print("======================================")
        print_pose(T_07)

    # --------------------------------------------------------
    # Fixed transformation from Joint 7 -> hand frame in the supplied model.
    #
    # Franka flange:
    #
    # a = 0
    # alpha = 0
    # d = 0.107 m
    # theta = -45 degrees (the hand's fixed local Z rotation)
    # --------------------------------------------------------

    T_7F = dh_transform(
        a=0.0,
        alpha=0.0,
        d=0.107,
        theta=-np.pi / 4
    )

    # Base -> Flange
    T_0F = T_07 @ T_7F

    if verbose:
        print("\n======================================")
        print("BASE -> FRANKA HAND")
        print("======================================")
        print_pose(T_0F)

    return T_0F


def heal_6dof(joint_angles_deg=None, verbose=True):
    """Heal end-effector FK from a standard DH table derived from its joint axes.

    The small angle offsets retain the instructor XML's rounded CAD Euler
    angles. The base and tool rotations account for the model's frame choices.
    The returned pose is in the MuJoCo world frame, in metres.
    """
    if joint_angles_deg is None:
        joint_angles_deg = [0] * 6
    if len(joint_angles_deg) != 6:
        raise ValueError("Heal needs six joint angles")
    q = np.deg2rad(joint_angles_deg)
    table = np.array([
        [0.0, 1.57079208, 0.14979963, q[0]],
        [0.3, np.pi, 0.0875, q[1] + np.pi / 2],
        [0.0, 1.57, 0.08737315, q[2] + 0.00079633],
        [0.0, 0.50951, 0.3773558, q[3] + np.pi],
        [0.0, 1.57079208, 0.00010061, q[4] - 1.57159265],
        [0.0, 0.0, -0.1227, q[5]],
    ])
    base = np.diag([-1.0, -1.0, 1.0, 1.0])
    base[:3, 3] = [0.0, 0.0, 0.171]
    tool = np.diag([-1.0, 1.0, -1.0, 1.0])
    pose = base @ forward_kinematics(table, verbose=verbose) @ tool
    if verbose:
        print("\nBASE -> HEAL END EFFECTOR")
        print_pose(pose)
    return pose


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("Forward Kinematics using DH Parameters")

    print("\nChoose robot:")
    print("1 -> Generic 6-DOF robot")
    print("2 -> Franka Research 3")
    print("3 -> Heal 6-DOF arm")

    choice = input("\nEnter choice: ")

    if choice == "1":
        generic_6dof()

    elif choice == "2":
        franka_7dof()

    elif choice == "3":
        heal_6dof()

    else:
        print("Invalid choice.")
