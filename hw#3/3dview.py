# -*- coding: utf-8 -*-
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from functools import reduce
import sys
import math

INF = 1e+100

# Объявляем все глобальные переменные
global xrot         # Величина вращения по оси x
global yrot         # Величина вращения по оси y
global ambient      # рассеянное освещение
global greencolor   # Цвет елочных иголок
global treecolor    # Цвет елочного стебля
global lightpos     # Положение источника освещения
global m

def get_parsed_line(file):
    res = file.readline()
    while res[0] == '#':
        res = file.readline()
    return res.replace('\n', '').split(' ');

class normal:
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

class point:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    def draw(self):
        glVertex(self.x, self.y, self.z)

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

class triangle:
    def __init__(self):
        self.norm = normal()
        self.points = [point() for i in range(3)]

    def draw(self):
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
                return True

class model:
    def __init__(self, filename = ''):
        self.triangles = []
        self.n = 0
        self.name = ''
        self.min = [INF] * 3
        self.max = [-INF] * 3
        if (filename != ''):
            self.load(filename)

    def set_min_max(self):
        self.min = reduce(lambda a, b: list(map(min, a, b)), \
                          [x.min() for x in self.triangles])
        self.max = reduce(lambda a, b: list(map(max, a, b)), \
                          [x.max() for x in self.triangles])

    def draw(self):
        for t in self.triangles:
            t.draw()

    def show(self):
        pass

    def load(self, filename):
        f = open(filename)
        line = get_parsed_line(f)
        if line[0] == 'solid':
            self.name = line[1]
            t = triangle()
            while (t.load(f)):
                self.triangles.append(t)
                t = triangle()
            self.n = len(self.triangles)
            print '%d triangles read' % self.n
            self.set_min_max()
            return True

        else:
            raise ValueError('Input file is supposed to be binary STL. Binary STL is not supported yet.')



# Процедура инициализации
def init(m):
    global xrot         # Величина вращения по оси x
    global yrot         # Величина вращения по оси y
    global ambient      # Рассеянное освещение
    global greencolor   # Цвет елочных иголок
    global treecolor    # Цвет елочного ствола
    global lightpos     # Положение источника освещения

    global scale

    xrot = 0.0                          # Величина вращения по оси x = 0
    yrot = 0.0                          # Величина вращения по оси y = 0
    ambient = (1.0, 1.0, 1.0, 10)        # Первые три числа цвет в формате RGB, а последнее - яркость
    greencolor = (0.2, 0.8, 0.0, 0.8)   # Зеленый цвет для иголок
    treecolor = (0.9, 0.6, 0.3, 0.8)    # Коричневый цвет для ствола
    lightpos = (m.max[0] + 1.0, m.max[1] + 1.0, m.max[2] + 1.0)          # Положение источника освещения по осям xyz
    scale = 1

    glClearColor(0.5, 0.5, 0.5, 1.0)                # Серый цвет для первоначальной закраски
    xmin, ymin, zmin = m.min
    xmax, ymax, zmax = m.max

    xmean = (xmax + xmin) / 2
    xd = xmax - xmin
    xmin = xmean - xd
    xmax = xmean + xd

    ymean = (ymax + ymin) / 2
    yd = ymax - ymin
    ymin = ymean - yd
    ymax = ymean + yd

    zmean = (zmax + zmin) / 2
    zd = zmax - zmin
    zmin = zmean - zd
    zmax = zmean + zd

    glOrtho(xmin - 1, xmax + 1, ymin - 1, ymax + 1, zmin - 1, zmax + 1)                # Определяем границы рисования по горизонтали и вертикали
    glRotatef(-90, 1.0, 0.0, 0.0)                   # Сместимся по оси Х на 90 градусов
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient) # Определяем текущую модель освещения
    glEnable(GL_LIGHTING)                           # Включаем освещение
    glEnable(GL_LIGHT0)                             # Включаем один источник света
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos)     # Определяем положение источника света
    #glEnable(GL_COLOR_MATERIAL)

# Процедура обработки специальных клавиш
def specialkeys(key, x, y):
    global xrot
    global yrot

    global scale

    #print key

    # Обработчики для клавиш со стрелками
    if key == GLUT_KEY_UP:      # Клавиша вверх
        xrot -= 2.0             # Уменьшаем угол вращения по оси Х
    if key == GLUT_KEY_DOWN:    # Клавиша вниз
        xrot += 2.0             # Увеличиваем угол вращения по оси Х
    if key == GLUT_KEY_LEFT:    # Клавиша влево
        yrot -= 2.0             # Уменьшаем угол вращения по оси Y
    if key == GLUT_KEY_RIGHT:   # Клавиша вправо
        yrot += 2.0             # Увеличиваем угол вращения по оси Y

    if key == 'w':
        scale *= 1.5
    if key == 's':
        scale /= 1.5

    glutPostRedisplay()         # Вызываем процедуру перерисовки


# Процедура перерисовки
def draw_model():
    global xrot
    global yrot
    global lightpos
    global greencolor
    global treecolor

    global scale

    glClear(GL_COLOR_BUFFER_BIT)                                # Очищаем экран и заливаем серым цветом
    glPushMatrix()                                              # Сохраняем текущее положение "камеры"

    glRotatef(xrot, 1.0, 0.0, 0.0)                              # Вращаем по оси X на величину xrot
    glRotatef(yrot, 0.0, 1.0, 0.0)                              # Вращаем по оси Y на величину yrot
    #glLightfv(GL_LIGHT0, GL_POSITION, lightpos)  # Источник света вращаем вместе с елкой

    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, treecolor)
    #glutSolidCylinder(100, 100, 20, 20)
    #glTranslatef(0, 0, 0)
    glBegin(GL_TRIANGLES)
    m.draw()
    glEnd()
    '''
    glTranslatef(dx, dy, dz)

    # Рисуем ствол елки
    # Устанавливаем материал: рисовать с 2 сторон, рассеянное освещение, коричневый цвет
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, treecolor)
    #glTranslatef(0.0, 0.0, -0.7)                                # Сдвинемся по оси Z на -0.7
    # Рисуем цилиндр с радиусом 0.1, высотой 0.2
    # Последние два числа определяют количество полигонов
    glutSolidCylinder(0.1, 0.2, 20, 20)
    '''

    '''
    # Рисуем ветки елки
    # Устанавливаем материал: рисовать с 2 сторон, рассеянное освещение, зеленый цвет
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, greencolor)
    glTranslatef(0.0, 0.0, 0.2)                                 # Сдвинемся по оси Z на 0.2
    # Рисуем нижние ветки (конус) с радиусом 0.5, высотой 0.5
    # Последние два числа определяют количество полигонов
    glutSolidCone(0.5, 0.5, 20, 20)
    glTranslatef(0.0, 0.0, 0.3)                                 # Сдвинемся по оси Z на -0.3
    glutSolidCone(0.4, 0.4, 20, 20)                             # Конус с радиусом 0.4, высотой 0.4
    glTranslatef(0.0, 0.0, 0.3)                                 # Сдвинемся по оси Z на -0.3
    glutSolidCone(0.3, 0.3, 20, 20)                             # Конус с радиусом 0.3, высотой 0.3
    '''
    glPopMatrix()                                               # Возвращаем сохраненное положение "камеры"
    glutSwapBuffers()                                           # Выводим все нарисованное в памяти на экран

def main():
    # Здесь начинается выполнение программы
    # Использовать двойную буферизацию и цвета в формате RGB (Красный, Зеленый, Синий)
    if (not bool(glutInitDisplayMode)):
        raise ValueError("No GLUT installed. Unable to find GLUT dll.")

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    # Указываем начальный размер окна (ширина, высота)
    glutInitWindowSize(300, 300)
    # Указываем начальное положение окна относительно левого верхнего угла экрана
    glutInitWindowPosition(50, 50)
    # Инициализация OpenGl
    glutInit(sys.argv)
    # Создаем окно с заголовком "Happy New Year!"
    glutCreateWindow(b"Happy New Year!")
    # Определяем процедуру, отвечающую за перерисовку
    glutDisplayFunc(draw_model)
    # Определяем процедуру, отвечающую за обработку клавиш
    glutSpecialFunc(specialkeys)
    glutKeyboardFunc(specialkeys)
    # Вызываем нашу функцию инициализации
    init(m)
    # Запускаем основной цикл
    glutMainLoop()


if (__name__ == '__main__'):
    global m
    m = model(sys.argv[1])
    print m.min
    print m.max
    main()