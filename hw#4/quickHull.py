import sys
import os
import numpy as np
from Queue import Queue
from itertools import combinations
#from itertools import reduce

sys.path.append(os.path.join('..', 'hw#3'))
#from view3d import triangle, main

INF = 1e+100

def scal_prod(a, b):
    return (a * b).sum()

def sign(x):
    if (abs(x) < 1e-9):
        return 0
    if (x < 0):
        return -1
    return 1

class pseudo_facet:
    def __init__(self, pts):
        self.pts = pts

class point:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z])
    def get_coords(self):
        return self.coords

edges_dict = {}

class edge:
    def __init__(self, p1, p2, order = 1):
        self.pts = np.array([p1, p2])
        self.order = order
        self.facets = set()
        global edges_dict
        key1, key2 = self.get_keys()
        if key1 in edges_dict.keys():
            edges_dict[key1].append(self)
        elif key2 in edges_dict.keys():
            edges_dict[key2].append(self)
        else:
            edges_dict[key1] = [self]

    def get_keys(self):
        p1 = self.pts[0]
        p2 = self.pts[1]
        key1 = (tuple(p1.get_coords()), tuple(p2.get_coords()))
        key2 = (tuple(p2.get_coords()), tuple(p1.get_coords()))
        return key1, key2

    def add_facet(self, f):
        self.facets.add(f)
    def get_facets(self):
        key1, key2 = self.get_keys()
        if key1 in edges_dict.keys():
            twins = edges_dict[key1]
        else:
            twins = edges_dict[key2]
        return reduce(lambda a, b: a | b, map(lambda x: x.facets, twins))

    def get_points(self):
        return self.pts
    def get_coords(self):
        return [p.get_coords() for p in self.pts[::self.order]]

class facet:
    facet_list = []
    def __init__(self, e1, e2, e3, parent):
        self.facet_list.append(self)
        ps1 = e1.get_points()
        ps2 = e2.get_points()
        ps3 = e3.get_points()
        e1.add_facet(self)
        e2.add_facet(self)
        e3.add_facet(self)
        assert ((ps1[1] == ps2[0]) and (ps2[1] == ps3[0]) and (ps3[1] == ps1[0]) and\
             (ps1[0] != ps2[0]) and (ps1[1] != ps2[1]) and
             (ps2[0] != ps3[0]) and (ps2[1] != ps3[1]) and
             (ps3[0] != ps1[0]) and (ps3[1] != ps1[1])), 'Error: invalid facet'
        self.edges = np.array([e1, e2, e3])
        self.calc_norm()
        self.pts = []
        self.choose_pts(parent.pts)
        self.is_actual = True

    def choose_pts(self, pts_list):
        self.pts = []
        self_pts = self.get_points()
        p0 = sum(self.get_coords()) / 3
        for p in pts_list:
            if p in self_pts:
                continue
            if scal_prod(p.get_coords() - p0, self.norm) >= 0:
                self.pts.append(p)

    def calc_norm(self):
        v1, v2, v3 = [p.get_coords() for p in self.get_points()]
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

    def get_norm(self):
        return self.norm
    def get_neignbours(self):
        return reduce(lambda a, b: a | b, map(lambda e: e.get_facets(), self.edges)) - set([self])
    def get_coords(self):
        return np.array(list(map(lambda x: x.get_coords()[0], self.edges)))
    def get_edges(self):
        return self.edges
    def get_points(self):
        return np.array(list(map(lambda x: x.get_points()[0], self.edges)))
    def get_farthest(self):
        if len(self.pts) > 0:
            return self.pts[np.argmax(map(lambda p: scal_prod(self.norm, p.get_coords() - self.get_coords()[0]), self.pts))]
        return None
    def perform_horizon(self, p):
        horizon = []
        neighbours = self.get_neignbours()
        self.is_actual = False
        for f in neighbours:
            if (f.is_actual) and (sign(scal_prod(sum(f.get_coords()) / 3, f.get_norm())) >= 0):
                horizon += f.perform_horizon(p)
            else:
                horizon.append(list(set(self.get_edges()) & set(f.get_edges()))[0])
        return horizon

def read_pts(filename):
    res = []
    with open(filename, 'r') as f:
        N = int(f.readline())
        for i in range(N):
            x, y, z = map(float, f.readline().split())
            res.append(point(x, y, z))
        f.close()
    return np.array(res)


def get_minmaxes(pts):
    pts = np.array([p.get_coords() for p in pts])
    return reduce(lambda a, b: (map(min, a[0], b), map(max, a[1], b)), pts, ([INF] * 3, [-INF] * 3))

def quickHull(filename):
    pts = read_pts(filename)
    minimaxes = np.array(get_minmaxes(pts))
    point_in_minimaxes = []
    for p in pts:
        coords = p.get_coords()
        point_in_minimaxes += [any([coords[i] in minimaxes[:, i] for i in range(3)])]
    ind_minimax = np.where(point_in_minimaxes)[0]
    assert len(ind_minimax) >= 3
    diffs_max = 0
    for inds in combinations(ind_minimax, 3):
        #some scoring to choose the best initial facet
        coords = map(lambda x: pts[x].get_coords(), inds)
        diffs = abs(np.array([coords[0] - coords[1], coords[0] - coords[2], coords[1] - coords[2]])).sum()
        if diffs > diffs_max:
            ind_best = inds
            diffs_max = diffs
    pts_minimax = [pts[ind_best[i]] for i in range(3)]
    edges_initial_0 = [edge(pts_minimax[0], pts_minimax[1]),
                     edge(pts_minimax[1], pts_minimax[2]),
                     edge(pts_minimax[2], pts_minimax[0])]
    edges_initial_1 = [edge(pts_minimax[0], pts_minimax[2]),
                       edge(pts_minimax[2], pts_minimax[1]),
                       edge(pts_minimax[1], pts_minimax[0])]

    ps = pseudo_facet(pts)
    f0 = facet(*edges_initial_0, parent=ps)
    f1 = facet(*edges_initial_1, parent=ps)

    Q = Queue()
    Q.put(f0)
    Q.put(f1)
    while not Q.empty():
        f = Q.get()
        pf = f.get_farthest()
        if pf == None:
            continue
        horizon = f.perform_horizon(pf)
        for e in horizon:
            p0 = e.get_points()[0]
            p1 = e.get_points()[1]
            Q.put(facet(e, edge(p1, pf), edge(pf, p0), e))

    hull = []
    for f in f0.facet_list:
        if f.is_actual:
            hull.append(f)
    for f in hull:
        print f.get_coords()

if __name__ == '__main__':
    quickHull(sys.argv[1])