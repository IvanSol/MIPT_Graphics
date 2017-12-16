import sys
import os
import numpy as np
from Queue import Queue
from itertools import combinations
#from itertools import reduce
import pickle
import random

global visualization_level

#sys.path.append(os.path.join('..', 'hw#3'))
#from view3d import visualize_model

INF = 1e+100

def scal_prod(a, b):
    return (a * b).sum()

def vect_prod(a, b):
    ax, ay, az = a
    bx, by, bz = b
    nx = float(ay * bz - az * by)
    ny = float(az * bx - ax * bz)
    nz = float(ax * by - ay * bx)
    return np.sqrt(nx * nx + ny * ny + nz * nz)


def sign(x):
    if (abs(x) < 1e-9):
        return 0
    if (x < 0):
        return -1
    return 1

class point:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z])
    def get_coords(self):
        return self.coords

class edge:
    def __init__(self, p1, p2):
        self.pts = np.array([p1, p2])
        self.facets = set()
    def add_facet(self, f):
        self.facets.add(f)
    def get_facets(self):
        return self.facets
    def get_points(self):
        return self.pts
    def get_coords(self):
        return [p.get_coords() for p in self.pts]

class oedge():
    def __init__(self, e, order):
        self.e = e
        self.order = order
    def add_facet(self, f):
        self.e.add_facet(f)
    def get_facets(self):
        return self.e.facets
    def get_points(self):
        return self.e.pts[::self.order]
    def get_coords(self):
        return [p.get_coords() for p in self.e.pts[::self.order]]


class facet:
    facet_list = []
    def __init__(self, e1, e2, e3, current_pts):
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
        self.choose_pts(current_pts)
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
    def get_neighbours(self):
        return reduce(lambda a, b: a | b, map(lambda e: e.get_facets(), self.get_edges())) - set([self])
    def get_coords(self):
        return np.array(list(map(lambda x: x.get_coords()[0], self.get_oedges())))
    def get_edges(self):
        return map(lambda x: x.e, self.edges)
    def get_oedges(self):
        return self.edges
    def get_points(self):
        return np.array(list(map(lambda x: x.get_points()[0], self.get_oedges())))
    def get_farthest(self):
        if len(self.pts) > 0:
            return self.pts[np.argmax(map(lambda p: scal_prod(self.norm, p.get_coords() - self.get_coords()[0]), self.pts))]
        return None
    def perform_horizon(self, p):
        horizon = []
        pts = []
        neighbours = self.get_neighbours()
        if sign(scal_prod(sum(self.get_coords()) / 3 - p.get_coords(), self.get_norm())) < 0:
            self.is_actual = False
            pts += self.pts
        else:
            return horizon, pts

        for f in neighbours:
            if (f.is_actual) and (sign(scal_prod(sum(f.get_coords()) / 3 - p.get_coords(), f.get_norm())) < 0):
                neigb_horizon, neighb_pts = f.perform_horizon(p)
                horizon += neigb_horizon
                pts += neighb_pts
            elif (f.is_actual):
                horizon_edges = list(set(self.get_edges()) & set(f.get_edges()))
                for he in horizon_edges:
                    for oe in self.get_oedges():
                        if oe.e == he:
                            horizon.append(oe)
                            break
        return horizon, pts

def read_pts(filename):
    res = []
    with open(filename, 'r') as f:
        N = int(f.readline())
        for i in range(N):
            x, y, z = map(float, f.readline().split())
            res.append(point(x, y, z))
        f.close()
    return np.array(res)

def read_pts_numpy(filename):
    with open(filename) as f:
        pts = pickle.load(f)
        f.close()
    pts = pts.reshape((pts.size / 3, 3))
    res = []
    for p in np.unique(pts, axis = 0):
        res.append(point(*p))
    return res


def get_minmaxes(pts):
    pts = np.array([p.get_coords() for p in pts])
    return reduce(lambda a, b: (map(min, a[0], b), map(max, a[1], b)), pts, ([INF] * 3, [-INF] * 3))

def visualize(pts, lns, facets):
    if facets != None:
        triangles = np.array([f.get_coords() for f in facets if f.is_actual])
    else:
        triangles = np.array([])
    if lns != None:
        lines = np.array([ln.get_coords() for ln in lns])
    else:
        lines = np.array([])
    points = np.array([p.get_coords() for p in pts])
    np_model = np.array([triangles, lines, points])
    pickle.dump(obj=np_model, file=open('visualize.pickle', 'w'))
    os.system('python ../hw#3/view3d.py -np ./visualize.pickle')
    #visualize_model()

def check_horizon_closed(horizon):
    if (len(horizon) < 2):
        return None
    hor_dict = {}
    for oe in horizon:
        key = oe.get_points()[0]
        if key in hor_dict.keys():
            if hor_dict[key] != oe:
                pass
                #print "Warning! Wrong horizon!"
                #return None
        hor_dict[key] = oe
    oriented_horizon = [horizon[0]]
    for i in range(len(horizon) - 1):
        oe0 = oriented_horizon[-1]
        key = oe0.get_points()[1]
        if key in hor_dict.keys():
            oriented_horizon.append(hor_dict[key])
        else:
            return None
    if oriented_horizon[0].get_points()[0] == oriented_horizon[-1].get_points()[1]:
        return oriented_horizon
    else:
        return None

def quickHull(filename):
    if filename.split('.')[1] == 'txt':
        pts = read_pts(filename)
    elif filename.split('.')[1] == 'pickle':
        pts = read_pts_numpy(filename)
    else:
        print 'Unknown file format'
        return
    random.seed(42)
    if visualization_level >= 2:
        visualize(pts, None, None)

    minimaxes = np.array(get_minmaxes(pts))
    point_in_minimaxes = []
    for p in pts:
        coords = p.get_coords()
        point_in_minimaxes += [any([coords[i] in minimaxes[:, i] for i in range(3)])]
    ind_minimax = np.where(point_in_minimaxes)[0]
    if len(ind_minimax) >= 3:
        random.shuffle(ind_minimax)
        diffs_max = 0
        k = 0
        for inds in combinations(ind_minimax, 3):
            #some scoring to choose the best initial facet
            coords = map(lambda x: pts[x].get_coords(), inds)
            diffs = abs(np.array([coords[0] - coords[1], coords[0] - coords[2], coords[1] - coords[2]])).sum()
            if diffs > diffs_max:
                ind_best = inds
                diffs_max = diffs
            k += 1
            if (k == 1000):
                break
        pts_minimax = [pts[ind_best[i]] for i in range(3)]
    else:
        s_max = 0
        pts_minimax_candidate = []
        for i in range(1000):
            pts_minimax_candidate = [pts[i] for i in ind_minimax]
            pts_minimax_candidate.append(random.choice(pts))
            s = vect_prod(pts_minimax_candidate[1].get_coords() - pts_minimax_candidate[0].get_coords(),
                          pts_minimax_candidate[2].get_coords() - pts_minimax_candidate[0].get_coords())
            if s > s_max:
                pts_minimax = pts_minimax_candidate
                s_max = s

    e0 = edge(pts_minimax[0], pts_minimax[1])
    e1 = edge(pts_minimax[1], pts_minimax[2])
    e2 = edge(pts_minimax[2], pts_minimax[0])
    edges_initial_0 = [oedge(e0, 1), oedge(e1, 1), oedge(e2, 1)]
    edges_initial_1 = [oedge(e2, -1), oedge(e1, -1), oedge(e0, -1)]

    f0 = facet(*edges_initial_0, current_pts=pts)
    f1 = facet(*edges_initial_1, current_pts=pts)

    if visualization_level >= 4:
        visualize(pts, None, f0.facet_list)

    Q = Queue()
    Q.put(f0)
    Q.put(f1)
    while not Q.empty():
        f = Q.get()
        if not f.is_actual:
            continue
        pf = f.get_farthest()
        if pf == None:
            continue
        horizon, current_pts = f.perform_horizon(pf)
        #print pf.get_coords()
        #for p in map(lambda x: x.get_coords(), horizon):
        #    print p
        #print
        horizon = check_horizon_closed(horizon)
        if horizon is None:
            print('Warning: wrong horizon!')
            continue
        if visualization_level >= 3:
            visualize(pts, horizon, f0.facet_list)

        assert horizon is not None, 'Error: invalid horizon'
        horizon_points = map(lambda x: x.get_points()[0], horizon)
        horizon_update_edges = map(lambda p: edge(p, pf), horizon_points)
        horizon_straight = map(lambda e: oedge(e, 1), horizon_update_edges[1:] + [horizon_update_edges[0]])
        horizon_reverse = map(lambda e: oedge(e, -1), horizon_update_edges)
        for e0, e1, e2 in zip(horizon, horizon_straight, horizon_reverse):
            Q.put(facet(e0, e1, e2, current_pts))
        if visualization_level >= 4:
            visualize(pts, None, f0.facet_list)

    hull = []
    for f in f0.facet_list:
        if f.is_actual:
            hull.append(f)
    if visualization_level >= 1:
        visualize(pts, None, hull)
    for f in hull:
        print f.get_coords()

if __name__ == '__main__':
    global visualization_level

    # level = 0 - show nothing
    # level = 1 - show just result
    # level = 2 - show initial points and result
    # level = 3 - show initial points, result and horizons
    # level = 4 - show initial points, result, horizons and hull after connecting new point to horizon.
    visualization_level = 2

    if (len(sys.argv) == 3):
        visualization_level = int(sys.argv[2])
    quickHull(sys.argv[1])