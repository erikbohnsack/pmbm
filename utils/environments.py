import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import cv2


def plot_bev_gt(labels, frame_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ego_vehicle = patches.Rectangle((-0.5, -2), 1, 4, color="blue", alpha=0.50)
    ax.add_patch(ego_vehicle)


    for l in labels[frame_idx]:
        if l.type[0] == 'DontCare':
            continue

        x_pos = l.location[0]
        z_pos = l.location[2]
        width = l.dimensions[1]
        length = l.dimensions[2]

        rot_y = l.rotation_y
        _x = x_pos - width / 2
        _z = z_pos - length / 2
        r = patches.Rectangle((_x, _z), width, length, color="red", alpha=0.2)
        # TODO: fix the rotation of bboxes
        #t = mpl.transforms.Affine2D().rotate_around(x_pos, z_pos, -rot_y) + ax.transData
        #r.set_transform(t)
        ax.add_patch(r)

        plt.text(x_pos-width,z_pos+length, str(l.track_id),color='black')
        plt.plot(x_pos, z_pos, 'r*')


    plt.xlim(-xlim, xlim)
    plt.ylim(-0.25*zlim, 1.75*zlim)
    plt.grid(True)

    plt.show()


def plot_image(image):
    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_gt_et(labels, frame_idx, estimated_targets):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ego_vehicle = patches.Rectangle((-0.5, -2), 1, 4, color="blue", alpha=0.50)
    ax.add_patch(ego_vehicle)

    for l in labels[frame_idx]:
        if l.type[0] == 'DontCare':
            continue

        x_pos = l.location[0]
        z_pos = l.location[2]
        width = l.dimensions[1]
        length = l.dimensions[2]

        rot_y = l.rotation_y
        _x = x_pos - width / 2
        _z = z_pos - length / 2
        r = patches.Rectangle((_x, _z), width, length, color="red", alpha=0.2)
        # TODO: fix the rotation of bboxes
        #t = mpl.transforms.Affine2D().rotate_around(x_pos, z_pos, -rot_y) + ax.transData
        #r.set_transform(t)
        ax.add_patch(r)

        plt.text(x_pos-width,z_pos+length, str(l.track_id),color='black')
        plt.plot(x_pos, z_pos, 'r*')
    if not len(estimated_targets) == 0:
        for est in estimated_targets:
            _x = est['single_target'].state[0]
            _z = est['single_target'].state[1]
            ax.plot(_x, _z, 'go', lw=2, ms=2)

    plt.xlim(-xlim, xlim)
    plt.ylim(-0.25*zlim, 1.75*zlim)
    plt.grid(True)

    plt.show()

