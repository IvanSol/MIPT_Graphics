import sys
import os
import numpy as np
from Queue import Queue
#from itertools import reduce

sys.path.append(os.path.join('..', 'hw#3'))
#from view3d import triangle, main

INF = 1e+100

def scal_prod(a, b):
    return (a * b).sum()

class pseudo_edge:
    def __init__(self, pts):
        self.pts = pts

class edge:
    def __init__(self, p0, p1, p2, parent):
        self.coords = np.array([p0, p1, p2])
        self.calc_norm()
        self.choose_pts(parent.pts)
        self.pts = []

    def choose_pts(self, pts_list):
        self.pts = []
        p0 = self.coords.sum() / 3
        for p in pts_list:
            if scal_prod(p - p0, self.norm) > 0:
                self.pts.append(p)

    def calc_norm(self):
        v1, v2, v3 = [p.get_coords() for p in self.coords]
        ax, ay, az = v2 - v1
        bx, by, bz = v3 - v1

        nx = float(ay * bz - az * by)
        ny = float(az * bx - ax * bz)
        nz = float(ax * by - ay * bx)
        norm2 = np.sqrt(nx * nx + ny * ny + nz * nz)

        nx /= norm2
        ny /= norm2
        nz /= norm2

        self.norm = np.array([nx, ny, nz])

    def get_coords(self):
        return np.array(map(lambda x: x.get_coords(), self.coords))

    def get_farthest(self):
        return self.pts[np.argmax(map(lambda p: scal_prod(self.norm, p - self.coords[0]), self.pts))]


class point:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z])

    def get_coords(self):
        return self.coords

def read_pts(filename):
    res = []
    with open(filename, 'r') as f:
        N = int(f.readline())
        for i in range(N):
            x, y, z = map(float, f.readline().split())
            res.append([x, y, z])
        f.close()
    return np.array(res)


def get_minmaxes(pts):
    return reduce(lambda a, b: (map(min, a[0], b), map(max, a[1], b)), pts, ([INF] * 3, [-INF] * 3))

def quickHull(filename):
    pts = read_pts(filename)
    minimaxes = np.array(get_minmaxes(pts))
    point_in_minimaxes = []
    for p in pts:
        point_in_minimaxes += [any([p[i] in minimaxes[:, i] for i in range(3)])]
    ind = np.where(point_in_minimaxes)[0][:3]
    assert len(ind) == 3
    pts_minimax = [point(*pts[ind[i]]) for i in range(3)]

    #Not tested:
    ps = pseudo_edge(pts)
    Q = Queue()
    Q.put(edge(*pts_minimax, ps))
    Q.put(edge(*pts_minimax[::-1], ps))
    while not Q.empty():
        e = Q.get()
        p0, p1, p2 = e.get_coords()
        pf = e.get_farthest()
        Q.put(edge(p0, p1, pf, e))
        Q.put(edge(p1, p2, pf, e))
        Q.put(edge(p2, p0, pf, e))

if __name__ == '__main__':
    quickHull(sys.argv[1])