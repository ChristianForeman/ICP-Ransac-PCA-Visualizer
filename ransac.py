#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import math
import rospy
from arc_utilities.ros_helpers import get_connected_publisher
from sensor_msgs.msg import PointCloud2


def get_normal(points):
    # Takes in 3 3d points as a 3x3 numpy array and returns a matrix for the plane
    # get two vectors which form a basis for the plane
    v1 = points[1, :] - points[0, :]
    v2 = points[2, :] - points[0, :]
    # Compute the cross product which is the normal vector to the plane
    cross = np.cross(v1, v2)
    # Return the norm
    return np.expand_dims(cross, axis=0).T


def main():
    rospy.init_node("ransac")
    ransac_pub = get_connected_publisher("fit", PointCloud2, queue_size=10)
    init_pub = get_connected_publisher("init", PointCloud2, queue_size=10)

    # Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')

    # Show the input point cloud
    fig = utils.view_pc([pc], marker='.')

    pc = np.array(pc)
    pc = np.squeeze(pc)

    # Fit a plane to the data using ransac
    iterations = 750
    threshold = 0.06

    best_inlier_count = 0
    best_norm = None
    point_in_best = None

    shuffled = pc
    for i in range(iterations):
        inliers = 0

        # Select three random points in the point cloud
        np.random.shuffle(shuffled)
        subset = shuffled[:3, :]

        cur_norm = get_normal(subset)

        # point_in_cur is one of the points in the plane
        point_in_cur = np.asmatrix(np.around(subset[:1, :].T, 2))

        # find d
        d = -point_in_cur.T * cur_norm

        temp = np.squeeze(cur_norm)
        # Count the number of inliers
        for j in range(pc.shape[0]):
            jth_point = pc[j, :]
            residual = np.abs(np.dot(temp, jth_point) + d) / math.sqrt(np.dot(temp, temp)) ** 2

            if residual < threshold:
                inliers += 1

        # Check if this fit is the best so far
        if inliers > best_inlier_count:
            print("iteration", i, "found an improvement with", inliers, "inliers.")
            best_inlier_count = inliers
            best_norm = cur_norm
            point_in_best = point_in_cur

    # Draw the fitted plane
    fig = utils.draw_plane(fig, normal=best_norm, pt=point_in_best, color=(0.1, 0.7, 0.1, 0.5),
                           length=[-0.75, 1], width=[-0.5, 1.2])

    # Create another point cloud that highlights points that are included in the threshold
    d = -point_in_best.T * best_norm

    within = []
    temp = np.squeeze(best_norm)
    # Count the number of inliers
    for j in range(pc.shape[0]):
        jth_point = pc[j, :]
        residual = np.abs(np.dot(temp, jth_point) + d) / math.sqrt(np.dot(temp, temp)) ** 2

        if residual < threshold:
            within.append(jth_point)

    within = np.asmatrix(np.vstack(within))
    within = utils.convert_matrix_to_pc(within.T)
    utils.view_pc([within], fig, 'r')

    # Show the resulting point cloud and plane
    plt.show()


if __name__ == '__main__':
    main()
