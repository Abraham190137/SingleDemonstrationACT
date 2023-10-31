import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def normalize(array:np.ndarray):
    assert np.linalg.norm(array) > 0, "Cannot normalize a vector of length 0."
    return array/np.linalg.norm(array)

# Functions for visualizing the warp matrix calculation
def graph_points_and_line(points:np.ndarray, start:np.ndarray = None, end:np.ndarray=None, new_start:np.ndarray = None, new_end:np.ndarray = None, title:str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if title is not None:
        ax.set_title(title)
    max_coord = np.max(points)
    min_coord = np.min(points)

    cmap = np.linspace(0, 1, points.shape[0])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=cmap)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(min_coord, max_coord)
    ax.set_ylim(min_coord, max_coord)
    ax.set_zlim(min_coord, max_coord)

    if not(start is None or end is None):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])
        ax.scatter(start[0], start[1], start[2], marker = 'o', color="blue", facecolors='none')
        ax.scatter(end[0], end[1], end[2], marker = 'x', color="blue")
        max_coord = max([max_coord, np.max(start), np.max(end)])
        min_coord = min([min_coord, np.min(start), np.min(end)])

    if not(new_start is None or new_end is None):
        ax.scatter(new_start[0], new_start[1], new_start[2], marker = 'o', color="black", facecolors='none')
        ax.scatter(new_end[0], new_end[1], new_end[2], marker = 'x', color="black")
        max_coord = max([max_coord, np.max(new_start), np.max(new_end)])
        min_coord = min([min_coord, np.min(new_start), np.min(new_end)])
    return fig, ax

def graph_rotated_axis(rotation_matricies:List[np.ndarray]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for rotation_matrix in rotation_matricies:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        x_axis_rotated = rotation_matrix @ np.array([1, 0, 0]).reshape(3, 1)
        y_axis_rotated = rotation_matrix @ np.array([0, 1, 0]).reshape(3, 1)
        z_axis_rotated = rotation_matrix @ np.array([0, 0, 1]).reshape(3, 1)
        
        for axis, color in zip([x_axis_rotated, y_axis_rotated, z_axis_rotated], ['r', 'g', 'b']):
            ax.plot([0, axis[0, 0]],
                    [0, axis[1, 0]],
                    [0, axis[2, 0]], color=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig, ax
    


def skew_symmetric(v:np.ndarray):
    """
    Returns the skew-symmetric matrix of a 3D vector.
    :param v: The 3D vector. Size: (3,1) or (3,)
    :return: The skew-symmetric matrix. Size: (3, 3)
    """
    v_flat = v.flatten()
    return np.array([[0, -v_flat[2], v_flat[1]], [v_flat[2], 0, -v_flat[0]], [-v_flat[1], v_flat[0], 0]])

def axis_and_angle_to_rotation_matrix(axis, angle):
    """
    Returns the rotation matrix for a rotation around an axis by a given angle.
    :param axis: The axis of rotation. Size: (3,1) or (3,)
    :param angle: The angle of rotation.
    """
    axis = np.copy(axis).reshape(3, 1)/np.linalg.norm(axis)
    return np.cos(angle)*np.eye(3) + skew_symmetric(axis) * np.sin(angle) + axis@axis.T * (1 - np.cos(angle))

def warp_matrix(rec_start, rec_end, new_start, new_end):
    """
    Generates the matrix to warp a recorded trajectory to a new start and end point.
    :param rec_start: The recorded start point. Size: (3,)
    :param rec_end: The recorded end point. Size: (3,)
    :param new_start: The new start point. Size: (3,)
    :param new_end: The new end point. Size: (3,)
    :return: The warp matrix. Size: (4, 4)
    """

    # To warp the recorded points, we need to perform a series of operations:
    # 1a. Rotate the recorded points so that the vector from the rec_start to the rec_end point is oriented
    #    the same direction as the vector from the new_start to the new_end point.
    # 1b. Rotate the recorded points so that the z-axis is oriented upwards (along the z axis in the original frame).
    # 2. Scale the points so that the distance between the origin and the recorded end
    #    point is the same as the distance between the origin and the new end point.
    # 3. Translate the points so that the recorded start point is at the same location as the new start point.

    # Calculate the direction vectors for rotation
    rec_direction:np.ndarray = (rec_end - rec_start).reshape(3, 1)
    new_direction:np.ndarray = (new_end - new_start).reshape(3, 1)

    # Calculate the rotation axis and angle
    rotation_angle:float = np.arccos(new_direction.T@rec_direction / (np.linalg.norm(new_direction) * np.linalg.norm(rec_direction)))

    if abs(rotation_angle) > 0.001: # to avoid numerical errors when the rotation angle is at or near 0.
        rotation_axis:np.ndarray = normalize(np.cross(rec_direction.flatten(), new_direction.flatten())).reshape(3, 1)

        # Calculate the rotation matrix
        first_rotation_matrix = axis_and_angle_to_rotation_matrix(rotation_axis, rotation_angle)
    else:
        first_rotation_matrix = np.eye(3)

    
    # Now rotate the trajectory along the transitory axis (from the start point to the end point) so that the z-axis is oriented upwards
    z_axis:np.ndarray = np.array([0, 0, 1]).reshape(3, 1)
    new_z_axis:np.ndarray= first_rotation_matrix @ z_axis # current z-axis

    # Project the new z-axis onto the plane perpendicular to the transitory axis:
    projected_new_z_axis:np.ndarray = new_z_axis - (new_z_axis.T@new_direction) * new_direction/np.linalg.norm(new_direction)**2

    # Goal z axis (z-axis in the original frame, projected onto the plane perpendicular to the transitory axis)
    projected_z_axis:np.ndarray = z_axis - (z_axis.T@new_direction) * new_direction/np.linalg.norm(new_direction)**2

    # if the projected z axis parellel to the transitory axis, then we don't need to rotate the trajectory again.
    if np.linalg.norm(projected_new_z_axis) < 1e-6:
        second_rotation_matrix = np.eye(3)
    else:
        # Figure out the angle between the goal and current z axes.
        second_rotation_angle = np.arccos(projected_z_axis.T@projected_new_z_axis/(np.linalg.norm(projected_z_axis)*np.linalg.norm(projected_new_z_axis)))

        if abs(second_rotation_angle) > 0.001: # to avoid numerical errors when the rotation angle is at or near 0.
            # The second rotation should be around the new_direction axis, but to determine the sign, we calcuated using the cross product.
            second_rotation_axis:np.ndarray = normalize(np.cross(projected_new_z_axis.flatten(), projected_z_axis.flatten())).reshape(3, 1)
            assert np.linalg.norm(np.cross(new_direction.flatten(), second_rotation_axis.flatten())) < 1e-6, "projected_z_axis and new_x_axis are not parallel, cross product is : {}".format(np.cross(new_direction.flatten(), second_rotation_axis.flatten()))

            # Get the second rotation matrix
            second_rotation_matrix = axis_and_angle_to_rotation_matrix(second_rotation_axis, second_rotation_angle)
        else:
            second_rotation_matrix = np.eye(3)


    # Combine the two rotation matrices to get the full rotation matrix, then turn it into a transformation matrix
    rotation_matrix = second_rotation_matrix @ first_rotation_matrix
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = rotation_matrix

    # Scale the points:
    # Calculate the scale factor
    scale_factor = np.linalg.norm(new_direction) / np.linalg.norm(rec_direction)
    scale_transform = np.array([[scale_factor, 0, 0, 0],
                                [0, scale_factor, 0, 0],
                                [0, 0, scale_factor, 0],
                                [0, 0, 0, 1]])


    # Calculate the translation vector
    transformed_rec_start = scale_transform @ rotation_transform @ np.array([rec_start[0], rec_start[1], rec_start[2], 1])
    translation_vector = new_start - transformed_rec_start[:3]
    translation_transform = np.array([[1, 0, 0, translation_vector[0]],
                                      [0, 1, 0, translation_vector[1]],
                                      [0, 0, 1, translation_vector[2]],
                                      [0, 0, 0, 1]])
    
    # Combine the three transforms into the overall warp matrix
    T_warp = translation_transform @ scale_transform @ rotation_transform

    return T_warp


def warp_trajectory(recorded_trajectory, new_start, new_end):
    """
    Warps a recorded trajectory to a new start and end point.
    :param recorded_trajectory: The recorded trajectory to be warped. Size: (N, 3)
    :param new_start: The new start point. Size: (3,)
    :param new_end: The new end point. Size: (3,)
    :return: The warped trajectory. Size: (N, 3)
    """

    # Calculate the warp matrix:
    T_warp = warp_matrix(recorded_trajectory[0, :], recorded_trajectory[-1, :], new_start, new_end)

    # Return the warped trajectory without the homogeneous component
    return T_warp @ np.concatenate((recorded_trajectory.T, np.ones((1, recorded_trajectory.shape[0]))), axis=0)[:3, :].T 


def visualize_warp_matrix_calculation(rec_start, rec_end, new_start, new_end, recorded_trajectory):
    """
    Similar to the warp matrix function, but visualizes the steps of the warp matrix calculation.
    Generates the matrix to warp a recorded trajectory to a new start and end point.
    :param rec_start: The recorded start point. Size: (3,)
    :param rec_end: The recorded end point. Size: (3,)
    :param new_start: The new start point. Size: (3,)
    :param new_end: The new end point. Size: (3,)
    :return: The warp matrix. Size: (4, 4)
    """

    # To warp the recorded points, we need to perform a series of operations:
    # 1a. Rotate the recorded points so that the vector from the rec_start to the rec_end point is oriented
    #    the same direction as the vector from the new_start to the new_end point.
    # 1b. Rotate the recorded points so that the z-axis is oriented upwards (along the z axis in the original frame).
    # 2. Scale the points so that the distance between the origin and the recorded end
    #    point is the same as the distance between the origin and the new end point.
    # 3. Translate the points so that the recorded start point is at the same location as the new start point.

    # Calculate the direction vectors for rotation
    rec_direction:np.ndarray = (rec_end - rec_start).reshape(3, 1)
    new_direction:np.ndarray = (new_end - new_start).reshape(3, 1)

    # Calculate the rotation axis and angle
    rotation_axis:np.ndarray = normalize(np.cross(rec_direction.flatten(), new_direction.flatten())).reshape(3, 1)
    rotation_angle:float = np.arccos(new_direction.T@rec_direction / (np.linalg.norm(new_direction) * np.linalg.norm(rec_direction)))

    # Calculate the rotation matrix
    first_rotation_matrix = axis_and_angle_to_rotation_matrix(rotation_axis, rotation_angle)
    
   # Now rotate the trajectory along the transitory axis (from the start point to the end point) so that the z-axis is oriented upwards
    z_axis:np.ndarray = np.array([0, 0, 1]).reshape(3, 1)
    new_z_axis:np.ndarray= first_rotation_matrix @ z_axis # current z-axis

    # Project the new z-axis onto the plane perpendicular to the transitory axis:
    projected_new_z_axis:np.ndarray = new_z_axis - (new_z_axis.T@new_direction) * new_direction/np.linalg.norm(new_direction)**2

    # Goal z axis (z-axis in the original frame, projected onto the plane perpendicular to the transitory axis)
    projected_z_axis:np.ndarray = z_axis - (z_axis.T@new_direction) * new_direction/np.linalg.norm(new_direction)**2

    # The second rotation should be around the new_direction axis, but to determine the sign, we calcuated using the cross product.
    second_rotation_axis:np.ndarray = normalize(np.cross(projected_new_z_axis.flatten(), projected_z_axis.flatten())).reshape(3, 1)
    assert np.linalg.norm(np.cross(new_direction.flatten(), second_rotation_axis.flatten())) < 1e-6, "projected_z_axis and new_x_axis are not parallel, cross product is : {}".format(np.cross(new_direction.flatten(), second_rotation_axis.flatten()))

    # Figure out the angle between the goal and current z axes.
    second_rotation_angle = np.arccos(projected_z_axis.T@projected_new_z_axis/(np.linalg.norm(projected_z_axis)*np.linalg.norm(projected_new_z_axis)))
    print('second_rotation_angle:', second_rotation_angle)

    # Get the second rotation matrix
    second_rotation_matrix = axis_and_angle_to_rotation_matrix(second_rotation_axis, second_rotation_angle)

    # Combine the two rotation matrices to get the full rotation matrix, then turn it into a transformation matrix
    rotation_matrix = second_rotation_matrix @ first_rotation_matrix
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = rotation_matrix

    # Scale the points:
    # Calculate the scale factor
    scale_factor = np.linalg.norm(new_direction) / np.linalg.norm(rec_direction)
    print('scale_factor: ', scale_factor)
    scale_transform = np.array([[scale_factor, 0, 0, 0],
                                [0, scale_factor, 0, 0],
                                [0, 0, scale_factor, 0],
                                [0, 0, 0, 1]])


    # Calculate the translation vector
    transformed_rec_start = scale_transform @ rotation_transform @ np.array([rec_start[0], rec_start[1], rec_start[2], 1])
    translation_vector = new_start - transformed_rec_start[:3]
    translation_transform = np.array([[1, 0, 0, translation_vector[0]],
                                      [0, 1, 0, translation_vector[1]],
                                      [0, 0, 1, translation_vector[2]],
                                      [0, 0, 0, 1]])
    
    # Combine the three transforms into the overall warp matrix
    T_warp = translation_transform @ scale_transform @ rotation_transform

    fig, ax = graph_rotated_axis([first_rotation_matrix])
    ax.plot([0, projected_z_axis[0, 0]], [0, projected_z_axis[1, 0]], [0, projected_z_axis[2, 0]], color='purple')
    
    fig, ax = graph_rotated_axis([rotation_matrix])
    ax.plot([0, projected_z_axis[0, 0]], [0, projected_z_axis[1, 0]], [0, projected_z_axis[2, 0]], color='purple')

    graph_points_and_line(recorded_trajectory, recorded_trajectory[0, :], recorded_trajectory[-1, :], new_start, new_end, title='Original Trajectory')
    rotated_points = rotation_transform @ np.concatenate((recorded_trajectory.T, np.ones((1, recorded_trajectory.shape[0]))), axis=0)
    scaled_points = scale_transform @ rotated_points
    warped_trajectory = translation_transform @ scaled_points

    graph_points_and_line(rotated_points[:3, :].T, rotated_points[:3, 0], rotated_points[:3, -1], new_start, new_end, title='Rotated Points')
    graph_points_and_line(scaled_points[:3, :].T, scaled_points[:3, 0], scaled_points[:3, -1], new_start, new_end, title='Scaled Points')
    graph_points_and_line(warped_trajectory[:3, :].T, warped_trajectory[:3, 0], warped_trajectory[:3, -1], new_start, new_end, title='Warped Trajectory')
    plt.show()

    return T_warp


def warp_matrix2D(rec_start:np.ndarray, rec_end:np.ndarray, new_start:np.ndarray, new_end:np.ndarray) -> np.ndarray:
    """
    Generates the matrix to warp a recorded trajectory to a new start and end point.
    :param rec_start: The recorded start point. Size: (2)
    :param rec_end: The recorded end point. Size: (2)
    :param new_start: The new start point. Size: (2)
    :param new_end: The new end point. Size: (2)
    :return: The warp matrix. Size: (3,3)
    """

    # To warp the recorded points, we need to prefrom 3 opeartions:
        # 1. Rotate the recorded points so that the vector form the rec_start to the rec_end point is orientated
        #    the same direction as the vector from the new_start to the new_end point.
        # 2. Scale the points so that the distance between the origin and the recorded end 
        #    point is the same as the distance between the origin and the new end point.
        # 3. Translate the points so that the recored start point is at the same location as the new start point.
    
    # Calculate the angle of rotation of the vector from the new_start to the new_end point:
    theta = np.arctan2(new_end[1] - new_start[1], new_end[0] - new_start[0])

    # Calculate the angle of rotation of the vector from the rec_start to the rec_end point:
    alpha = np.arctan2(rec_end[1] - rec_start[1], rec_end[0] - rec_start[0])

    # Calculate the angle of rotation needed to rotate the recorded points to the new points orientation:
    beta = theta - alpha
    rotation_matrix = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

    # Tranformation matrix that scales the recorded end point to the new end point:
    scale_factor = np.sqrt((new_end[0] - new_start[0])**2 + (new_end[1] - new_start[1])**2)/np.sqrt((rec_end[0] - rec_start[0])**2 + (rec_end[1] - rec_start[1])**2)
    scale_matrix = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])

    # Tranformation matrix that translates the recorded start point to the new start point:
    rotated_and_scaled_rec_start = scale_matrix@rotation_matrix@np.array([rec_start[0], rec_start[1], 1])
    translation_matrix = np.array([[1, 0, new_start[0] - rotated_and_scaled_rec_start[0]], 
                                   [0, 1, new_start[1] - rotated_and_scaled_rec_start[1]], 
                                   [0, 0, 1]])

    # Finally, create the overall transformation matrix to warp the recorded trajectory:
    # Steps: Convert to frame O, translate to origin, rotate to x-axis, scale to new end point, convert back to world frame
    T_warp = translation_matrix@scale_matrix@rotation_matrix

    return T_warp

def warp_trajectory2D(recorded_trajectory:np.ndarray, new_start:np.ndarray, new_end:np.ndarray) -> np.ndarray:
    """
    Warps a recorded trajectory to a new start and end point.
    :param recorded_trajectory: The recorded trajectory to be warped. Size: (N,2)
    :param new_start: The new start point. Size: (2)
    :param new_end: The new end point. Size: (2)
    :return: The warped trajectory. Size: (N,2)warp_trajectory
    """
    
    # Calculate the warp matrix:
    T_warp = warp_matrix(recorded_trajectory[0, :], recorded_trajectory[-1, :], new_start, new_end)

    # Warp the recorded trajectory:
    homogeneous_recorded_trajectory = np.ones((len(recorded_trajectory), 3))
    homogeneous_recorded_trajectory[:, 0:2] = recorded_trajectory
    warped_trajectory = T_warp@homogeneous_recorded_trajectory.T

    return warped_trajectory[0:2, :].T # Return the warped trajectory without the homogeneous component


def main():
    recorded_x = np.linspace(0, 2*np.pi, 20)
    recorded_y = np.zeros(recorded_x.shape)
    recorded_z = np.sin(recorded_x)
    
    new_start = np.array([0, 0, 0])
    new_start = np.random.uniform(-5, 5, 3)
    new_goal = np.random.uniform(-5, 5, 3)

    test_rec_start = np.array([-0.08, -0.01,  0.22])
    test_rec_end = np.array([0.1 , 0.1 , 0.02])
    test_new_start = np.array([0.00552648, 0.23263776, 0.45029313])
    test_new_end = np.array([ 0.02466098, -0.11006149,  0.02])

    T_warp = warp_matrix(test_rec_start, test_rec_end, test_new_start, test_new_end)
    recorded_trajectory = np.array([test_rec_start, test_rec_end])
    visualize_warp_matrix_calculation(test_rec_start, test_rec_end, test_new_start, test_new_end, recorded_trajectory)

    # recorded_trajectory = np.array([recorded_x, recorded_y, recorded_z]).T
    # print(recorded_trajectory)
    # T_warp = warp_matrix(recorded_trajectory[0, :], recorded_trajectory[-1, :], new_start, new_goal)
    # warped_trajectory = visualize_warp_matrix_calculation(recorded_trajectory[0, :], recorded_trajectory[-1, :], new_start, new_goal, recorded_trajectory)

if __name__ == "__main__":
    main()