# following easy writeup at http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

import numpy as np
import math

# Q  = center of circle as a 2-element numpy array
# r  = radius of circle as a scalar
# P1 = start of line segment as a 2-element numpy array
# P2 = end of line segment as a 2-element numpy array
# V  = P2 - P1 = vector for line segment


def intersect(P1, P2, Q, r):
    V = P2 - P1
    a = V.dot(V)
    b = 2.0 * V.dot(P1 - Q)
    c = P1.dot(P1) + Q.dot(Q) - 2.0 * P1.dot(Q) - r**2


    disc = b**2 - 4.0 * a * c
    if disc < 0:
        #print "no intersection"
        return False, None

    #print "could be intersection..."
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)

    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        #print "would intersect if line was extended"
        return False, None
   
    #t = max(0, min(1, - b / (2.0 * a)))
    
    # Choose whichever t1 or t2 is closer
    if (0 <= t1 <= 1 and t1 < t2):
        t = t1
    else:
        t = t2

    #print "yes there is an intersection!"
    pt = P1 + t * V
    return True, pt