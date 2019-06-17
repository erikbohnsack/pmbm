import unittest
import platform
from .kitti_stuff import Kitti

if platform.system() == 'Darwin':
    root = '/Users/erikbohnsack/data'
else:
    root = '/home/mlt/data'


class TestKitti(unittest.TestCase):
    def test_imu(self):
        sequence_id = 0
        kitti = Kitti(root)
        imud = kitti.load_imu(sequence_id)
        print(imud)
