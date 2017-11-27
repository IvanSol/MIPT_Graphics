# -*- coding: utf-8 -*-
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from functools import reduce
import sys
import math
import numpy as np
from struct import unpack

INF = 1e+100

# Объявляем все глобальные переменные
global greencolor   # Цвет елочных иголок
global treecolor    # Цвет елочного стебля
global lightpos     # Положение источника освещения
global m
global color
global ambient  # рассеянное освещение
perspective_angle = 45.0
perspective = True
camera_params_saved=[
{
    'scale': -3.0,
    'xrot': 0.0,  # Величина вращения по оси x = 0
    'yrot': 0.0  # Величина вращения по оси y = 0
},
{
    'scale': -3.0,
    'xrot': 90.0,  # Величина вращения по оси x = 0
    'yrot': 0.0  # Величина вращения по оси y = 0
}]
while (len(camera_params_saved) < 10):
    camera_params_saved.append(dict(camera_params_saved[0]))
saving_mode = False
camera_params = camera_params_saved[0]

color1 = (0.9, 0.6, 0.3)
color2 = (1, 1, 1)
color2 = (1, 1, 1)

def get_parsed_line(file):
    res = file.readline()
    while res[0] == '#':
        res = file.readline()
    return res.replace('\n', '').split(' ');

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

    def draw(self, color=0):
        #glColor3f(*color)
        glVertex3f(self.x, self.y, self.z)

    def get_coords(self, eps = None):
        if eps:
            numbers = int(-math.log10(eps) + eps)
            return round(self.x, numbers), round(self.y, numbers), round(self.z, numbers)
        else:
            return self.x, self.y, self.z

    def normalize(self, v_min, v_max):
        self.x -= (v_min[0] + v_max[0]) / 2
        self.x /= (v_max[0] - v_min[0])
        self.y -= (v_min[1] + v_max[1]) / 2
        self.y /= (v_max[1] - v_min[1])
        self.z -= (v_min[2] + v_max[2]) / 2
        self.z /= (v_max[2] - v_min[2])

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
        '''
        print ('Triangle:')
        print('%d %d %d' % (self.points[0].x, self.points[0].y, self.points[0].z))
        print('%d %d %d' % (self.points[1].x, self.points[1].y, self.points[1].z))
        print('%d %d %d' % (self.points[2].x, self.points[2].y, self.points[2].z))
        print('Normal:')
        print('%f %f %f' % (nx, ny, nz))
        print
        '''
        return np.array([nx, ny, nz])

    def draw(self, mode, color):
        #color = np.random.uniform(0, 1, 3)
        if mode == 'lines':
            self.points[0].draw(color)
            self.points[1].draw(color)
            self.points[1].draw(color)
            self.points[2].draw(color)
            self.points[2].draw(color)
            self.points[0].draw(color)
        else:
            glNormal3f(self.norm.dx, self.norm.dy, self.norm.dz)
            for p in self.points:
                p.draw(color)

    def min(self):
        return [min(map(lambda x: x.x, self.points)),
                min(map(lambda x: x.y, self.points)),
                min(map(lambda x: x.z, self.points))]
    def max(self):
        return [max(map(lambda x: x.x, self.points)),
                max(map(lambda x: x.y, self.points)),
                max(map(lambda x: x.z, self.points))]

    def normalize(self, v_min, v_max):
        for p in self.points:
            p.normalize(v_min, v_max)

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
    def __init__(self, filename = ''):
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
        if (filename != ''):
            self.load(filename)

    def get_coords(self, eps = None):
        return [t.get_coords(eps) for t in self.triangles]

    def set_min_max(self):
        self.min = reduce(lambda a, b: list(map(min, a, b)), \
                          [x.min() for x in self.triangles])
        self.max = reduce(lambda a, b: list(map(max, a, b)), \
                          [x.max() for x in self.triangles])

    def normalize(self, mode = 'uniform'):
        norm_min = self.min
        norm_max = self.max
        abs_min = min(norm_min)
        abs_max = max(norm_max)
        if mode == 'uniform':
            norm_min = [abs_min] * len(norm_min)
            norm_max = [abs_max] * len(norm_max)
        for t in self.triangles:
            t.normalize(norm_min, norm_max)
        self.eps /= abs_max - abs_min
        #print 'EPS = %.10f' % self.eps
        #print 'EPS = %.10f' % self.eps


    def calc_eps(self):
        coords = np.array(self.get_coords(1e-9))
        coords = coords.reshape((coords.size / 3, 3))
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


    def draw(self, mode, color):
        for t in self.triangles:
            t.draw(mode, color)

    def draw_shape(self):
        global camera_params
        scale = camera_params['scale']
        xrot = camera_params['xrot']
        yrot = camera_params['yrot']

        x_ang = xrot / 180 * math.pi
        y_ang = yrot / 180 * math.pi
        camera_pos = np.array([-scale * math.cos(x_ang) * math.sin(y_ang),
                               scale * math.sin(x_ang),
                               scale * math.cos(x_ang) * math.cos(y_ang)])

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

def do_perspective_if_needed():
    global perspective_angle
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if perspective:
        gluPerspective(perspective_angle, 1, 0.001, 100)
    else:
        glOrtho(-1, 1, -1, 1, -5, 5)


# Процедура инициализации
def init():
    global xrot         # Величина вращения по оси x
    global yrot         # Величина вращения по оси y
    global scale
    global ambient      # Рассеянное освещение
    global greencolor   # Цвет елочных иголок
    global treecolor    # Цвет елочного ствола
    global lightpos     # Положение источника освещения
    global color
    global camera_params
    global saving_mode
    global camera_params_saved
    global perspective

    ambient = (1.0, 1.0, 1.0, 1)        # Первые три числа цвет в формате RGB, а последнее - яркость
    greencolor = (0.2, 0.8, 0.0, 0.8)   # Зеленый цвет для иголок
    treecolor = (0.9, 0.6, 0.3, 0.8)    # Коричневый цвет для ствола
    #lightpos = (10.0, 10.0, 10.0)          # Положение источника освещения по осям xyz
    color = treecolor

    glClearColor(0.5, 0.5, 0.5, 1.0)                # Серый цвет для первоначальной закраски

    #glRotatef(-90, 1.0, 0.0, 0.0)                   # Сместимся по оси Х на 90 градусов

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
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

# Процедура обработки специальных клавиш
def is_a_number(chr):
    return (chr >= '0') and (chr <= '9')

def specialkeys(key, x, y):
    global camera_params
    global saving_mode
    global camera_params_saved
    global perspective
    global perspective_angle
    #print key

    if key == 'q':
        saving_mode = not saving_mode
    if is_a_number(key):
        key = int(key)
        if not saving_mode:
            if key < len(camera_params_saved):
                camera_params = dict(camera_params_saved[key])
        else:
            camera_params_saved[key] = dict(camera_params)
            saving_mode = False

    if not saving_mode:
        # Обработчики для клавиш со стрелками
        rotation_delta = 10.0

        if key == GLUT_KEY_UP:      # Клавиша вверх
            camera_params['xrot'] -= rotation_delta             # Уменьшаем угол вращения по оси Х
        if key == GLUT_KEY_DOWN:    # Клавиша вниз
            camera_params['xrot'] += rotation_delta             # Увеличиваем угол вращения по оси Х
        if key == GLUT_KEY_LEFT:    # Клавиша влево
            camera_params['yrot'] -= rotation_delta             # Уменьшаем угол вращения по оси Y
        if key == GLUT_KEY_RIGHT:   # Клавиша вправо
            camera_params['yrot'] += rotation_delta             # Увеличиваем угол вращения по оси Y

        if (key == 'w'):
            perspective_angle -= 1
            do_perspective_if_needed()
        if (key == 's'):
            perspective_angle += 1
            do_perspective_if_needed()

        if (key == '+'):
            camera_params['scale'] += 0.2
        if (key == '-'):
            camera_params['scale'] -= 0.2
        if key == 'x':
            m.to_draw_lines = not m.to_draw_lines
        if key == 'b':
            m.to_draw_surface = not m.to_draw_surface
        if key == 'v':
            m.to_draw_shape = not m.to_draw_shape
        if key == 'c':
            m.line_width += 1
        if key == 'z':
            if m.line_width > 1:
                m.line_width -= 1
        if key == 'p':
            perspective = not perspective
            do_perspective_if_needed()
    glutPostRedisplay()         # Вызываем процедуру перерисовки

# Процедура перерисовки
def draw_model():
    global lightpos
    global greencolor
    global treecolor
    global camera_params
    global saving_mode

    xrot = camera_params['xrot']
    yrot = camera_params['yrot']
    scale = camera_params['scale']

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

    glPushMatrix()                                              # Сохраняем текущее положение "камеры"
    # Очищаем экран и заливаем серым цветом
    #glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 0))  # Источник света вращаем вместе с елкой
    if perspective:
        glTranslatef(0, 0, scale)
    glRotatef(xrot, 1.0, 0.0, 0.0)                              # Вращаем по оси X на величину xrot
    glRotatef(yrot, 0.0, 1.0, 0.0)                              # Вращаем по оси Y на величину yrot
    #glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 0, 0))
    #glutSolidCylinder(1, 1, 100, 100)
    #glTranslatef(0, 0, 0)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)

    #glMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (.25, .25, .25, 1.0))
    #glMaterial(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, .5))
    #glMaterial(GL_FRONT, GL_SHININESS, (128.0, ))
    #glMaterial(GL_BACK, GL_AMBIENT_AND_DIFFUSE, (0., 0., 0., 0.))
    #glMaterial(GL_BACK, GL_SPECULAR, (0., 0., 0., 0.))
    #glMaterial(GL_BACK, GL_SHININESS, (0.,))

    #glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, treecolor)

    if m.to_draw_surface:
        glColor4f(0, 0.5, 0.5, 0)
        glBegin(GL_TRIANGLES)
        m.draw(mode='triangles', color = color1)
        glEnd()
    #glPolygonMode(GL_FRONT, GL_LINE)
    #glPolygonMode(GL_BACK, GL_LINE)
    #'''
    if m.to_draw_lines:
        glColor3f(0, 1.0, 0)
        glLineWidth(m.line_width)
        glBegin(GL_LINES)
        m.draw(mode='lines', color = color2)
        glEnd()

    if m.to_draw_shape:
        glColor3f(1.0, 0, 0)
        glLineWidth(m.line_width)
        glBegin(GL_LINES)
        m.draw_shape()
        glEnd()
    #'''
    #glDisable(GL_LIGHT0)
    glPopMatrix()                                               # Возвращаем сохраненное положение "камеры"
    glutSwapBuffers()                                           # Выводим все нарисованное в памяти на экран

def main():
    # Здесь начинается выполнение программы
    # Использовать двойную буферизацию и цвета в формате RGB (Красный, Зеленый, Синий)
    if (not bool(glutInitDisplayMode)):
        raise ValueError("No GLUT installed. Unable to find GLUT dll.")

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    # Указываем начальный размер окна (ширина, высота)
    glutInitWindowSize(640, 480)
    # Указываем начальное положение окна относительно левого верхнего угла экрана
    glutInitWindowPosition(50, 50)
    # Инициализация OpenGl
    glutInit()
    # Создаем окно с заголовком "Happy New Year!"
    glutCreateWindow(None)
    # Определяем процедуру, отвечающую за перерисовку
    glutDisplayFunc(draw_model)
    # Определяем процедуру, отвечающую за обработку клавиш
    glutSpecialFunc(specialkeys)
    glutKeyboardFunc(specialkeys)
    # Вызываем нашу функцию инициализации
    init()
    draw_model()
    # Запускаем основной цикл
    glutMainLoop()


if (__name__ == '__main__'):
    global m
    m = model(sys.argv[1])
    #print m.min
    #print m.max
    main()