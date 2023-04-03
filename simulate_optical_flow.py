# Import necessary libraries
import numpy as np
import cv2

focal_length = 100
# Create an empty image of specified size
h = 500
w = 800
image = np.full((h, w, 1), 255, dtype=np.uint8)

# convert euler angles to rotation vector
def euler_to_rvec(roll, pitch, yaw):
    """
    Converts Euler angles to a rotation vector
    
    Args:
    roll: float representing the roll angle in radians
    pitch: float representing the pitch angle in radians
    yaw: float representing the yaw angle in radians
    
    Returns:
    numpy array of shape (3,) representing the rotation vector
    """
    # Convert Euler angles to rotation matrix
    R = cv2.Rodrigues(np.array([roll, pitch, yaw]))[0]

    # Convert rotation matrix to rotation vector
    rvec = cv2.Rodrigues(R)[0].flatten()

    return rvec

# project points on an image with given motion
def project_points_on_imgae(image, points_3d, rvec, tvec):
    """
    Simulates point trajectories on an image when moving camera, with 6dof camera motion and 3d points coordinates as input
    
    Args:
    image: numpy array representing the image
    points_3d: numpy array of shape (n, 3) representing the 3D coordinates of n points
    camera_motion: numpy array of shape (6,) representing the 6 degrees of freedom camera motion
    
    Returns:
    numpy array of shape (n, 2) representing the 2D coordinates of n points on the image
    """
    # Define camera matrix
    camera_matrix = np.array([[focal_length, 0, image.shape[1]/2],
                              [0, focal_length, image.shape[0]/2],
                              [0, 0, 1]])
    
    # Project 3D points onto 2D image plane
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None) 

    return points_2d

# simulate point trajectories
def simulate_point_trajectories(image, num_points, rvec, tvec, output_img_path):
    # Simulate 3d points
    # Generate random 3D points within a range
    points_3d = np.random.uniform(low=-100, high=100, size=(num_points, 3))
    
    # Note: simulate point on a vertical plane
    for p in points_3d:
        p[2] = 30.0

    # project initial 2d points before motion
    init_rvec=np.zeros(3)
    init_tvec=np.zeros(3)
    points_2d_1 = project_points_on_imgae(image, points_3d, init_rvec, init_tvec)

    # projection 2d points after motion
    points_2d_2 = project_points_on_imgae(image, points_3d, rvec, tvec)

    # Draw lines between two set of 2d points under two frames
    for i in range(len(points_2d_1)):
        x1, y1 = points_2d_1[i].ravel()
        x2, y2 = points_2d_2[i].ravel()
        if x1 < 0 or x1 >= w or x2 < 0 or x2 >= w or y1 < 0 or y1 >= h or y2 < 0 or y2 >= h :
            continue
        cv2.circle(image, (int(x1), int(y1)), 5, (0, 0, 0), -1)
        cv2.circle(image, (int(x2), int(y2)), 5, (0, 0, 0), 1)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

    cv2.imwrite(output_img_path, image)

if __name__ == "__main__":
    num_points = 200
    rvec=np.zeros(3)
    tvec=np.zeros(3)

    # forward
    tvec=np.zeros(3)
    tvec[2] = 10.0
    # rvec = euler_to_rvec(0.0, 0.0, 0.0)
    simulate_point_trajectories(image, num_points, rvec, tvec, "optical_flow_forward.png")

    # back
    # tvec=np.zeros(3)
    # tvec[2] = -10.0

    # left
    # tvec=np.zeros(3)
    # tvec[0] = 10.0

    # up
    # tvec=np.zeros(3)
    # tvec[1] = -10.0