#!/usr/bin/env python
import utils
import numpy as np
import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from sensor_msgs.msg import PointCloud2

# error = []

def find_correspondences(p, q):
    # Returns a 3xn matrix of 3d points that are in an order such that p[i] and c[i] match to be a correspondence
    # overall_error = 0  # Can be used for viewing the error for each iteration, currently taken out
    remaining = np.copy(q)
    c = np.zeros_like(p)
    for i in range(p.shape[1]):
        # Find the remaining closest point
        best_index = None
        best_dist = np.inf
        for j in range(remaining.shape[1]):
            cur_dist = np.sqrt((p.item((0, i)) - remaining.item((0, j))) ** 2 +
                               (p.item((1, i)) - remaining.item((1, j))) ** 2 +
                               (p.item((2, i)) - remaining.item((2, j))) ** 2)
            if cur_dist < best_dist:
                best_index = j
                best_dist = cur_dist

        # Add in the new closest point
        c[:, i] = q[:, best_index]

        # overall_error += best_dist

    # overall_error /= p.shape[1]
    return c


def main():
    rospy.init_node("icp")
    icp_pub = get_connected_publisher("fit", PointCloud2, queue_size=10)
    init_pub = get_connected_publisher("init", PointCloud2, queue_size=10)
    target_pub = get_connected_publisher("target", PointCloud2, queue_size=10)

    iterations = 20

    # Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')
    pc_target = utils.load_pc('cloud_icp_target1.csv')  # Change this to load in a different target

    p = utils.convert_pc_to_matrix(pc_source)
    q = utils.convert_pc_to_matrix(pc_target)

    msg = utils.points_to_pc2_msg(np.array(p.T), "world")
    init_pub.publish(msg)

    msg = utils.points_to_pc2_msg(np.array(q.T), "world")
    target_pub.publish(msg)

    q_mean = np.sum(q, axis=1) / q.shape[1]
    # Randomly shuffle the columns (points) of the target ensuring we don't know the order
    q = q[:, np.random.permutation(q.shape[1])]

    for i in range(iterations):
        # Send the current guess to rviz
        msg = utils.points_to_pc2_msg(np.array(p.T), "world")
        icp_pub.publish(msg)

        # Estimate the correspondences between the two data sets
        c = find_correspondences(p, q)
        # c is a matrix of the same data as q but reordered in a way where the column should be a correspondence with p

        p_mean = np.sum(p, axis=1) / p.shape[1]

        x = p - p_mean
        y = c - q_mean

        # Get the covariance matrix
        s = np.matmul(x, y.T)

        # SVD
        u, s, vt = np.linalg.svd(s)

        temp = np.identity(3)
        temp[2][2] = np.linalg.det(np.matmul(vt.T, u.T))

        # Find the rotation matrix and the translation
        r = np.matmul(vt.T, np.matmul(temp, u.T))
        t = q_mean - np.matmul(r, p_mean)

        # Update the new source
        p = np.matmul(r, p) + t

        rospy.sleep(0.2)

    # The below section is for plotting error
    # p_new_pc = utils.convert_matrix_to_pc(p)
    #
    # utils.view_pc([pc_source, pc_target, p_new_pc], None, ['b', 'r', 'g'], ['.', '^', 'o'])
    # ax = plt.gca()
    # set_axes_equal(ax)
    # plt.axis([-0.15, 0.15, -0.15, 0.15])
    #
    # # Plot the error vs the iteration
    # x = list(range(iterations))
    # plt.figure(2)
    # # plotting the points
    # plt.plot(x, error)
    #
    # x = list(range(iterations))
    # plt.xlabel("Iteration")
    # plt.ylabel("MSE")
    # plt.title("Iteration vs Error for ICP")
    # plt.plot(x, error)
    #
    # plt.show()


if __name__ == '__main__':
    main()
