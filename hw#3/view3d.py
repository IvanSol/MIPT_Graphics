# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from functools import reduce
import sys
import os
import math
import numpy as np
from struct import unpack
import pickle

INF = 1e+100
scale = -3
user_pos_default = np.array([0, 0, scale])

def Quaternion_from_angle(angle, x, y, z):
    v = np.array([x, y, z])
    v /= (v*v).sum()
    x, y, z = v
    angle /= 2
    w = math.cos(angle)
    x = x * math.sin(angle)
    y = y * math.sin(angle)
    z = z * math.sin(angle)
    return Quaternion(w, x, y, z)

class Quaternion:
    def __init__(self, w, x, y, z):
        self.coords = np.array([w, x, y, z], dtype=np.float64)

    def rotate_vector(self, v):
        v_q = Quaternion(0, *v)
        res = self * v_q * self.get_inverted()
        return res.coords[1:]

    def get_camera_pos(self):
        return self.get_inverted().rotate_vector(user_pos_default)

    def apply(self):
        angle = math.acos(self.coords[0]) * 2 / math.pi * 180
        glRotate(angle, *self.coords[1:])

    def len(self):
        return math.sqrt((self.coords * self.coords).sum())

    def norm(self):
        l = self.len()
        if abs(l) > 1e-8:
            self.coords /= l

    def invert(self):
        self.coords[1:] = -self.coords[1:]
        self.norm()

    def get_inverted(self):
        inv = Quaternion(*self.coords)
        inv.coords[1:] = -inv.coords[1:]
        inv.norm()
        return inv

    def __mul__(self, other):
        w0, x0, y0, z0 = self.coords
        w1, x1, y1, z1 = other.coords
        return Quaternion(w0*w1 - x0*x1 - y0*y1 - z0*z1,
                          w0*x1 + x0*w1 + y0*z1 - z0*y1,
                          w0*y1 - x0*z1 + y0*w1 + z0*x1,
                          w0*z1 + x0*y1 - y0*x1 + z0*w1)

    def rotate_horizontal(self, delta):
        horizontal_axis = self.get_inverted().rotate_vector(np.array([0, 1, 0]))
        self.coords = (self * Quaternion_from_angle(delta, *horizontal_axis)).coords

    def rotate_vertical(self, delta):
        vertical_axis = self.get_inverted().rotate_vector(np.array([1, 0, 0]))
        self.coords = (self * Quaternion_from_angle(delta, *vertical_axis)).coords

def load_camera_params(i):
    global camera_params
    global camera_params_saved
    if camera_params_saved[i] != {}:
        camera_params = dict(camera_params_saved[i])
        camera_params['Q'] = Quaternion(*camera_params_saved[i]['Q'].coords)

def save_camera_params(i):
    global camera_params
    global camera_params_saved
    camera_params_saved[i] = dict(camera_params)
    camera_params_saved[i]['Q'] = Quaternion(*camera_params['Q'].coords)

# Global variables:
global m
crazy_mode = False
camera_params_saved=[
{
    'perspective': True,
    'perspective_angle': 45.0,
    'Q': Quaternion_from_angle(0, 1, 0, 0)
},
{
    'perspective': True,
    'perspective_angle': 45.0,
    'Q': Quaternion_from_angle(math.pi / 2, 1, 0, 0)
},
{
    'perspective': True,
    'perspective_angle': 45.0,
    'Q': Quaternion_from_angle(math.pi / 2, 0, 1, 0)
}
] + [{}] * 7
while (len(camera_params_saved) < 10):
    camera_params_saved.append(dict(camera_params_saved[0]))
saving_mode = False
global camera_params
load_camera_params(0)
show_axis = False
pair_mode = False
w = 1920
h = 1080

def get_parsed_line(file):
    res = file.readline()
    while res[0] == '#':
        res = file.readline()
    return res.replace('\n', '').split(' ')

class normal:
    binary_len = 12
    def __init__(self):
        self.dx = 0
        self.dy = 0
        self.dz = 0

    def load(self, input, mode = 'ASCII'):
        if mode == 'ASCII':
            if input[0] != 'normal':
                raise ValueError('STL file corrupted. Expected normal definition. Found: "%s"' % ' '.join(input))
            elif len(input) != 4:
                raise ValueError('STL file corrupted. Expected normal definition. Found: "%s"' % ' '.join(input))
            else:
                self.dx, self.dy, self.dz = map(float, input[1:])
                return True
        else:
            data = input #just renaming
            self.dx, self.dy, self.dz = unpack("fff", data)
            return True

    def as_vector(self):
        return np.array([self.dx, self.dy, self.dz])

class point:
    binary_len = 12

    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

    def as_vector(self):
        return np.array([self.x, self.y, self.z])

    def draw(self):
        glVertex3f(self.x, self.y, self.z)

    def get_coords(self, eps = None):
        if eps:
            numbers = int(-math.log10(eps) + eps)
            return round(self.x, numbers), round(self.y, numbers), round(self.z, numbers)
        else:
            return self.x, self.y, self.z

    def normalize(self, scale, mean):
        self.x = (self.x - mean[0]) * scale[0]
        self.y = (self.y - mean[1]) * scale[1]
        self.z = (self.z - mean[2]) * scale[2]

    def from_numpy(self, np_inp):
        self.x, self.y, self.z = np_inp

    def load(self, file, mode = 'ASCII'):
        if mode == 'ASCII':
            line = get_parsed_line(file)
            if line[0] != 'vertex':
                raise ValueError('STL file corrupted. Expected point definition. Found: "%s"' % ' '.join(line))
            elif len(line) != 4:
                raise ValueError('STL file corrupted. Expected point definition. Found: "%s"' % ' '.join(line))
            else:
                self.x, self.y, self.z = map(float, line[1:])
                return True
        else:
            data = file #just renaming
            self.x, self.y, self.z = unpack("fff", data)
            return True

class triangle:
    binary_len = 50
    def __init__(self):
        self.norm = normal()
        self.points = [point() for i in range(3)]

    def get_coords(self, eps = None):
        return [p.get_coords(eps) for p in self.points]

    def get_edges(self):
        return [(self.points[i],
                 self.points[(i + 1) % len(self.points)])
                 for i in range(len(self.points))]

    def calcNormal(self):
        v1, v2, v3 = [p.as_vector() for p in self.points]
        ax, ay, az = v2 - v1
        bx, by, bz = v3 - v1

        nx = float(ay * bz - az * by)
        ny = float(az * bx - ax * bz)
        nz = float(ax * by - ay * bx)
        norm2 = np.sqrt(nx * nx + ny * ny + nz * nz)

        nx /= norm2
        ny /= norm2
        nz /= norm2

        return np.array([nx, ny, nz])

    def draw(self, mode):
        if crazy_mode:
            color = np.random.uniform(0, 1, 3)
            glColor3f(*color)
        if mode == 'lines':
            self.points[0].draw()
            self.points[1].draw()
            self.points[1].draw()
            self.points[2].draw()
            self.points[2].draw()
            self.points[0].draw()
        else:
            glNormal3f(self.norm.dx, self.norm.dy, self.norm.dz)
            for p in self.points:
                p.draw()

    def min(self):
        return [min(map(lambda x: x.x, self.points)),
                min(map(lambda x: x.y, self.points)),
                min(map(lambda x: x.z, self.points))]
    def max(self):
        return [max(map(lambda x: x.x, self.points)),
                max(map(lambda x: x.y, self.points)),
                max(map(lambda x: x.z, self.points))]

    def normalize(self, scale, mean):
        for p in self.points:
            p.normalize(scale, mean)

    def from_numpy(self, np_inp):
        for i in range(np_inp.shape[0]):
            self.points[i].from_numpy(np_inp[i])
        self.norm.dx, self.norm.dy, self.norm.dz = self.calcNormal()

    def load(self, file, mode = 'ASCII'):
        if mode == 'ASCII':
            line = get_parsed_line(file)
            if line[0] == 'endsolid':
                return 0
            elif line[0] != 'facet':
                raise ValueError('STL file corrupted. Expected facet definition. Found: "%s"' % ' '.join(line))
            else:
                self.norm.load(line[1:], mode)
                line = get_parsed_line(file)
                if line != ['outer', 'loop']:
                    raise ValueError('STL file corrupted. Expected "outer loop". Found: "%s"' % ' '.join(line))

                for p in self.points:
                    p.load(file, mode)

                line = get_parsed_line(file)
                if line != ['endloop']:
                    raise ValueError('STL file corrupted. Expected "endloop". Found: "%s"' % ' '.join(line))

                line = get_parsed_line(file)
                if line != ['endfacet']:
                    raise ValueError('STL file corrupted. Expected "endfacet". Found: "%s"' % ' '.join(line))
                #return True
        else:
            data = file #just renaming
            self.norm.load(data[:self.norm.binary_len], mode)
            pos = self.norm.binary_len
            for i in range(3):
                self.points[i].load(data[pos : pos + self.points[i].binary_len], mode)
                pos += self.points[i].binary_len
        self.norm.dx, self.norm.dy, self.norm.dz = self.calcNormal()
        return True

def edge_rev(edge):
    return (edge[1], edge[0])

def scalar_product(v1, v2):
    return (v1 * v2).sum()

def scalar_product_cos(v1, v2):
    return scalar_product(v1, v2) / scalar_product(v1, v1) / scalar_product(v2, v2)

def sign(x):
    if (abs(x) < 1e-8):
        return 0
    if (x > 0):
        return 1
    return -1

class model:
    def __init__(self, filename = '', np_input=None):
        self.eps = 1e-2
        self.triangles = []
        self.n = 0
        self.to_draw_surface = True
        self.to_draw_lines = False#True
        self.to_draw_shape = False
        self.line_width = 2.0
        self.name = ''
        self.min = [INF] * 3
        self.max = [-INF] * 3
        self.edges_dict = {}
        self.pts = []
        self.lns = []
        if (filename != ''):
            self.load(filename)
        elif np_input is not None:
            self.from_numpy(np_input)

    def get_coords(self, eps = None):
        return [t.get_coords(eps) for t in self.triangles]

    def numpy_pickle(self, fn):
        f = open(fn, 'w')
        pickle.dump(np.array(self.get_coords()), f)
        f.close()

    def set_min_max(self):
        coords_t = np.array(self.get_coords(1e-9))
        coords_t = coords_t.reshape((coords_t.size / 3, 3))
        coords_p = np.array(map(lambda x: x.get_coords(1e-9), self.pts))
        coords_p = coords_p.reshape((coords_p.size / 3, 3))
        coords_l = np.array(map(lambda x: x.get_coords(1e-9), self.lns))
        coords_l = coords_l.reshape((coords_l.size / 3, 3))
        coords = np.concatenate([coords_t, coords_p, coords_l])
        self.min = [coords[:, i].min() for i in range(3)]
        self.max = [coords[:, i].max() for i in range(3)]
        '''
        self.min = reduce(lambda a, b: list(map(min, a, b)), \
                          [x.min() for x in self.triangles])
        self.max = reduce(lambda a, b: list(map(max, a, b)), \
                          [x.max() for x in self.triangles])
        '''

    def normalize(self, mode = 'uniform'):
        norm_min = np.array(self.min)
        norm_max = np.array(self.max)
        scale = norm_max - norm_min
        mean = (norm_max + norm_min) / 2
        if mode == 'uniform':
            scale = 1.0 / np.repeat(scale.max(), len(scale))
        else:
            scale = 1.0 / scale

        for t in self.triangles:
            t.normalize(scale, mean)
        for l in self.lns:
            l.normalize(scale, mean)
        for p in self.pts:
            p.normalize(scale, mean)
        self.eps *= scale.max()
        #print 'EPS = %.10f' % self.eps
        #print 'EPS = %.10f' % self.eps


    def calc_eps(self):
        coords_t = np.array(self.get_coords(1e-9))
        coords_t = coords_t.reshape((coords_t.size / 3, 3))
        coords_p = np.array(map(lambda x: x.get_coords(1e-9), self.pts))
        coords_p = coords_p.reshape((coords_p.size / 3, 3))
        coords_l = np.array(map(lambda x: x.get_coords(1e-9), self.lns))
        coords_l = coords_l.reshape((coords_l.size / 3, 3))
        coords = np.concatenate([coords_t, coords_p, coords_l])

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        x = np.sort(list(set(x)))
        y = np.sort(list(set(y)))
        z = np.sort(list(set(z)))
        dx = abs(x[:-1] - x[1:]).min()
        dy = abs(y[:-1] - y[1:]).min()
        dz = abs(z[:-1] - z[1:]).min()
        self.eps = min([dx, dy, dz]) / 10


    def draw(self, mode):
        for t in self.triangles:
            t.draw(mode)

    def draw_pts(self):
        for p in self.pts:
            p.draw()

    def draw_lines(self):
        for p in self.lns:
            p.draw()

    def draw_shape(self):
        global camera_params
        Q = camera_params['Q']
        camera_pos = Q.get_camera_pos()

        if self.edges_dict == {}:
            triangles_with_edges = [(t.get_edges(), t) for t in self.triangles]
            for t in triangles_with_edges:
                for edge_pts in t[0]:
                    edge = (edge_pts[0].get_coords(eps = self.eps),
                            edge_pts[1].get_coords(eps = self.eps))
                    if (edge_rev(edge) in self.edges_dict.keys()):
                        edge = edge_rev(edge)
                    #optimizable:
                    if (edge in self.edges_dict.keys()):
                        self.edges_dict[edge] += [(t[1], edge_pts)]
                    else:
                        self.edges_dict[edge] = [(t[1], edge_pts)]

        shape_list = []
        for e, points_n_triangle_list in self.edges_dict.iteritems():
            visibility = []
            border = []
            #if len(triangle_list) < 2:
            #    raise ValueError('Edge has only one triangle! The figure seems to be incomplete.')
            for t, p01 in points_n_triangle_list:
                mid = reduce(lambda a, b: a + b.as_vector(), t.points, np.zeros(3)) / 3 + camera_pos
                sgn = sign(scalar_product(mid, t.norm.as_vector()))
                #print t.get_coords(),
                #print '->', sgn
                if sgn < 0:
                    visibility += [True]
                    border += [False]
                elif sgn > 0:
                    visibility += [False]
                    border += [False]
                else:
                    visibility += [False]
                    border += [True]
            if (not all(visibility)) and any(visibility):
                shape_list += [p01[0], p01[1]]
        #print "Shape_size = %d" % (len(shape_list) / 2)
        for p in shape_list:
            p.draw()

    def show(self):
        pass

    def from_numpy(self, inp):
        self.name = 'Numpy model'
        tr_inp = inp[0]
        ln_inp = inp[1]
        pt_inp = inp[2]

        #assert tr_inp.shape[1] == 3, 'Unknown numpy shape'
        self.n = tr_inp.shape[0]
        for i in range(self.n):
            t = triangle()
            t.from_numpy(tr_inp[i])
            self.triangles.append(t)

        self.lns = [point(ln_inp[i][j][0], ln_inp[i][j][1], ln_inp[i][j][2]) for i in range(ln_inp.shape[0]) for j in range(2)]
        self.pts = [point(*pt_inp[i]) for i in range(pt_inp.shape[0])]

        print '%d triangles read' % self.n
        self.calc_eps()
        self.set_min_max()
        self.normalize()

    def load(self, filename):
        mode = ''
        f = open(filename, 'r')
        try:
            line = get_parsed_line(f)
            if line[0] == 'solid':
                #ASCII MODE:
                self.name = line[1]
                t = triangle()
                while (t.load(f)):
                    self.triangles.append(t)
                    t = triangle()
                self.n = len(self.triangles)
                mode = 'ASCII'
                #return True
        finally:
            f.close()

        if mode == '':
            try:
                f = open(filename, 'rb')
                data = f.read()
                f.close()
                self.n = unpack("I", data[80:84])[0]
                pos = 84
                for i in range(self.n):
                    t = triangle()
                    t.load(data[pos : pos + t.binary_len], mode='binary')
                    pos += t.binary_len
                    self.triangles.append(t)
                mode = 'binary'
            except:
                self.triangles = []
                raise ValueError('Error while reading binary file')

        print '%d triangles read' % self.n
        self.calc_eps()
        self.set_min_max()
        self.normalize()
        return mode != ''
        #else:
        #    raise ValueError('Input file is supposed to be binary STL. Binary STL is not supported yet.')

    def draw_me(self):
        if show_axis:
            glLineWidth(3)
            glBegin(GL_LINES)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(1, 0, 0)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 1, 0)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 1)
            glEnd()

        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)

        if self.to_draw_surface:
            glColor4f(0, 0.5, 0.5, 0)
            glBegin(GL_TRIANGLES)
            m.draw(mode='triangles')
            glEnd()

        glColor3f(0, 0, 0)
        glBegin(GL_POINTS)
        m.draw_pts()
        glEnd()

        glLineWidth(4)
        glColor3f(1, 0, 0)
        glBegin(GL_LINES)
        m.draw_lines()
        glEnd()

        if self.to_draw_lines:
            glColor3f(0, 1.0, 0)
            glLineWidth(m.line_width)
            glBegin(GL_LINES)
            m.draw(mode='lines')
            glEnd()

        if self.to_draw_shape:
            glColor3f(1.0, 0, 0)
            glLineWidth(m.line_width)
            glBegin(GL_LINES)
            m.draw_shape()
            glEnd()


def do_perspective_if_needed(move_ortho = False):
    global camera_params
    perspective_angle = camera_params['perspective_angle']
    perspective = camera_params['perspective']
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glViewport(0, 0, w, h)
    if perspective:
        gluPerspective(perspective_angle, (float(w)) / h, 0.001, 100)
    else:
        rate_w = float(w) / (640 / 2)
        rate_h = float(h) / (480 / 2)
        if move_ortho:
            glOrtho(-0.5 * rate_w, 2.5 * rate_w, -1.5 * rate_h, 1.5 * rate_h, -10, 10)
        else:
            glOrtho(-1 * rate_w, 1 * rate_w, -1 * rate_h, 1 * rate_h, -10, 10)


# Процедура инициализации
def init():
    global camera_params
    global saving_mode
    global camera_params_saved
    global perspective
    glClearColor(0.5, 0.5, 0.5, 1.0)
    BRIGHT4f = (1.0, 1.0, 1.0, 10.0)  # Color for Bright light
    DIM4f = (.2, .2, .2, 1.0)        # Color for Dim light

    brightLightPosition4f = (1, 0, 1, 0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, BRIGHT4f)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, BRIGHT4f)
    glLightfv(GL_LIGHT0, GL_POSITION, brightLightPosition4f)

    #glLightfv(GL_LIGHT1, GL_AMBIENT, DIM4f)
    #glLightfv(GL_LIGHT1, GL_DIFFUSE, DIM4f)
    #glLightfv(GL_LIGHT1, GL_POSITION, (-10, -10, -10, 1))
    #glEnable(GL_LIGHT1)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

    do_perspective_if_needed()

    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)  # Ensure farthest polygons render first
    glEnable(GL_NORMALIZE)  # Prevents scale from affecting color
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glPointSize(4)

def is_a_number(chr):
    return (chr >= '0') and (chr <= '9')

def specialkeys(key, x, y):
    global camera_params
    global saving_mode
    global camera_params_saved
    global crazy_mode
    global show_axis
    global pair_mode
    if key in ['B', 'С', 'V', 'Z', 'A', 'Q', 'E', 'W', 'S']:
        key = key.lower()
    #print key
    if key in ['s']:
        saving_mode = not saving_mode
    if is_a_number(key):
        key = int(key)
        if not saving_mode:
            if key < len(camera_params_saved):
                load_camera_params(key)
        else:
            save_camera_params(key)
            saving_mode = False

    if not saving_mode:
        # Обработчики для клавиш со стрелками
        rotation_delta = math.pi / 20

        if key == GLUT_KEY_UP:      # Клавиша вверх
            camera_params['Q'].rotate_vertical(rotation_delta)
        if key == GLUT_KEY_DOWN:    # Клавиша вниз
            camera_params['Q'].rotate_vertical(-rotation_delta)
        if key == GLUT_KEY_LEFT:    # Клавиша влево
            camera_params['Q'].rotate_horizontal(rotation_delta)
        if key == GLUT_KEY_RIGHT:   # Клавиша вправо
            camera_params['Q'].rotate_horizontal(-rotation_delta)

        if (key == '+'):
            camera_params['perspective_angle'] -= 1
            do_perspective_if_needed()
        if (key == '-'):
            camera_params['perspective_angle'] += 1
            do_perspective_if_needed()

        if key == 'c':
            m.to_draw_lines = not m.to_draw_lines
        if key == 'b':
            m.to_draw_surface = not m.to_draw_surface
        if key == 'v':
            m.to_draw_shape = not m.to_draw_shape
        if key == 'x':
            m.line_width += 1
        if key == 'z':
            if m.line_width > 1:
                m.line_width -= 1
        if key == 'q':
            camera_params['perspective'] = not camera_params['perspective']
            do_perspective_if_needed()
        if key == 'w':
            pair_mode = not pair_mode
        if key == 'a':
            show_axis = not show_axis
        if key == 'e':
            crazy_mode = not crazy_mode
        if key == ' ':
            pickle.dump(camera_params, open('camera.pickle', 'w'))
            glutLeaveMainLoop()
    glutPostRedisplay()

def draw_model():
    global camera_params
    global saving_mode
    global show_axis

    Q = camera_params['Q']

    glMatrixMode(GL_MODELVIEW)
    #glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    if saving_mode:
        s1 = 'Saving mode'
        s2 = 'Choose number from 0 to 9 to save camera position'
        glColor3f(0, 0, 0)
        #glDisable(GL_TEXTURE_2D)
        glRasterPos3f(-0.008, 0.0, -0.1)
        for c in s1:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(c))
        glRasterPos3f(-0.03, -0.005, -0.1)
        for c in s2:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(c))
        glutSwapBuffers()
        return


    glPushMatrix()
    if pair_mode:
        glTranslate(-1, 0, 0)
    if camera_params['perspective']:
        glTranslatef(0, 0, scale)
    do_perspective_if_needed()
    glMatrixMode(GL_MODELVIEW)
    Q.apply()
    m.draw_me()
    glPopMatrix()

    if pair_mode:
        glPushMatrix()
        glTranslate(1, 0, 0)
        camera_params['perspective'] = not camera_params['perspective']
        if camera_params['perspective']:
            glTranslatef(0, 0, scale)
        do_perspective_if_needed()
        glMatrixMode(GL_MODELVIEW)
        Q.apply()
        m.draw_me()
        glPopMatrix()
        camera_params['perspective'] = not camera_params['perspective']

    glutSwapBuffers()

def Reshape(width, height):
    global w
    global h
    w = width
    h = height

def visualize_model(model_numpy):
    global m
    m = model(np_input=model_numpy)
    main()

def main():
    # Использовать двойную буферизацию и цвета в формате RGB (Красный, Зеленый, Синий)
    if (not bool(glutInitDisplayMode)):
        raise ValueError("No GLUT installed. Unable to find GLUT dll.")
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    # Указываем начальный размер окна (ширина, высота)
    glutInitWindowSize(w, h)
    # Указываем начальное положение окна относительно левого верхнего угла экрана
    glutInitWindowPosition(0, 0)
    # Инициализация OpenGl
    glutInit()
    # Запускаем основной цикл
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
    # Создаем окно с заголовком - именем файла
    winid = glutCreateWindow(sys.argv[1])
    draw_model()
    # Определяем процедуру, отвечающую за перерисовку
    glutDisplayFunc(draw_model)
    # Определяем процедуру, отвечающую за обработку клавиш
    glutSpecialFunc(specialkeys)
    glutSpecialFunc(specialkeys)
    glutKeyboardFunc(specialkeys)
    glutReshapeFunc(Reshape)
    # Вызываем нашу функцию инициализации
    init()

    glutMainLoop()
    glutDestroyWindow(winid)


if (__name__ == '__main__'):
    global m
    if sys.argv[1] == '-np':
        global camera_params

        if os.path.isfile('camera.pickle'):
            f = open('camera.pickle')
            camera_params = pickle.load(f)
            f.close()
        model_numpy = pickle.load(open(sys.argv[2]))
        m = model(np_input=model_numpy)
    else:
        m = model(sys.argv[1])
    main()