'''
Author: Christopher Mannes

Unit testing code for mainQ1.py.
'''

import sys
import mainQ1
import unittest
from mainQ1 import *

class MainQ1Test(unittest.TestCase):

    def test_get_distance_for_tenth_degree_change(self):
        # Function signature
        # def get_distance(lat1, lon1, lat2, lon2)
        mainOutput = mainQ1.get_distance(-49.3,-72.8, -49.3,-72.9)
        expectedOutput = 3830 # meters
        self.assertEquals(expectedOutput, mainOutput)


if __name__ == "__main__":

    unittest.main()
