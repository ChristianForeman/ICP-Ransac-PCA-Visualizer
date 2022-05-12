#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from sensor_msgs.msg import PointCloud2


def main():
    rospy.init_node("pca")
    pca_pub = get_connected_publisher("fit", PointCloud2, queue_size=10)
    flatten_pub = get_connected_publisher("flatten", PointCloud2, queue_size=10)
    rot_pub = get_connected_publisher("rot", PointCloud2, queue_size=10)
    init_pub = get_connected_publisher("init", PointCloud2, queue_size=10)

    # Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    threshold = 0.0001

    # Show the input point cloud
    fig = utils.view_pc([pc])

    pc = utils.convert_pc_to_matrix(pc)
    msg = utils.points_to_pc2_msg(np.array(pc.T), "world")
    init_pub.publish(msg)

    original_pc = np.copy(pc)

    # Estimate the covariance matrix
    means = np.sum(pc, axis=1) / pc.shape[1]
    X = pc - means
    Q = np.matmul(X, X.T) / (pc.shape[1] - 1) # This is the estimated covar matrix

    # Perform SVD
    # s is a list of the singular values
    # vh is an orthonormal matrix
    u, s, vh = np.linalg.svd(Q)

    # Rotate the points to align with the XY plane
    X_new = np.matmul(vh, X)
    X_new_pc = utils.convert_matrix_to_pc(X_new)

    # Show the resulting point cloud
    fig = utils.view_pc([X_new_pc], fig, 'r')
    msg = utils.points_to_pc2_msg(np.array(X_new.T), "world")
    rot_pub.publish(msg)

    # List of indices of rows to remove
    to_remove = []
    # Rotate the points to align with the XY plane AND eliminate the noise
    for i in range(pc.shape[0]):
        variance = s[i] ** 2

        # If the eigenvalue of Q is less than the threshold, remove the dimension
        if variance < threshold:
            to_remove.append(i)

    vh_new = np.copy(vh)

    # TODO: Need to check if this is correct. Says to delete a row/column, but there needs to be something to replace it.
    for i in to_remove:
        vh_new[i, :] = 0

    # vh_new = np.delete(vh, to_remove, axis=0)

    X_new = np.matmul(vh_new, X)
    X_new_pc = utils.convert_matrix_to_pc(X_new)
    msg = utils.points_to_pc2_msg(np.array(X_new.T), "world")
    flatten_pub.publish(msg)

    # Show the resulting point
    fig = utils.view_pc([X_new_pc], fig, 'g')

    fig = utils.draw_plane(fig, np.matrix([0, 0, 1]).T, X_new[:, 0], (0.1, 0.7, 0.1, 0.5), [-1, 1.5], [-1, 1])

    new_norm = np.matmul(vh.T, np.matrix([0, 0, 1]).T)
    point_on_plane = np.sum(original_pc, axis=1) / pc.shape[1]  # TODO: need to find a way to reverse a point on the 2d plane
    # this way assumes
    # that a point on the plane is just at the mean location of all the points in the point cloud which does not work
    # well for extreme outliers

    fig = utils.draw_plane(fig, new_norm, point_on_plane, (0.1, 0.2, 0.1, 0.5), [-1, 1.5], [-1, 1])

    # Reverse matrix to see the fit on the data. Probably should take 3 points in X_new and plot
    # X_fit = np.matmul(inv(vh), X_new)
    # X_fit_pc = utils.convert_matrix_to_pc(X_fit)
    # fig = utils.view_pc([X_fit_pc], fig, 'g')



    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
