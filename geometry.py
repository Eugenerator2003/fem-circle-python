import numpy as np

current_type = float

class CoordinateFunction():
    def __init__(self, coord1, coord2):
        self.coord1 = coord1
        self.coord2 = coord2
        self.f = lambda t : (coord1 - coord2 ) * t + coord2

    def __call__(self, array_list):
        return self.f(array_list)

def get_param_f(coord1, coord2):
    return CoordinateFunction(coord1, coord2)

def get_line_f(x, y):
    return get_param_f(x[0], x[1]), get_param_f(y[0], y[1])

def get_all_points(fx, fy, t):
    return fx(t), fy(t)

def get_intermediate_points(fx, fy, t):
    t = t[1:len(t) - 1]
    return fx(t), fy(t)

def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_distance_from_line(fx, fy):
    x1, y1 = fx(0), fy(0)
    x2, y2 = fx(1), fy(1)
    return get_distance(x1, y1, x2, y2)

def get_distance_from_points(p1, p2):
    return get_distance(p1.x, p1.y, p2.x, p2.y)

class PlaneLine():
    def __init__(self, x1, y1, x2, y2):
        self.x = x1
        self.v = np.array([x1 - x2, y1 - y2])
        self.fx = CoordinateFunction(x1, x2)
        self.fy = CoordinateFunction(y1, y2)

        x_diff = x1 - x2
        y_diff = y1 - y2
        self.k = np.true_divide(y_diff, x_diff)
        self.c = - self.k * x2 + y2
    
    def __call__(self, t):
        return self.fx(t), self.fy(t)
    
    def get_y_by_x(self, x):
        return self.k * x + self.c
    
    def get_x_by_y(self, y):
        return (y + self.c) / self.k
    
    def get_line_distance(self):
        return get_distance_from_line(self.fx, self.fy)

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = 0

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, self.__class__) and 
               getattr(other, 'x', None) == self.x and
               getattr(other, 'y', None) == self.y)
    
    def __hash__(self) -> int:
        return hash(str(self.x) + str(self.y)) 

    def __str__(self) -> str:
        return f'({np.round(self.x, 3)}; {np.round(self.y, 3)})'

class Element():
    @staticmethod
    def get_x_y(element, old_points=True, for_fill=False, move=False):
        x = []
        y = []
        pts = element.points if old_points else (element.displaced if not move else element.displaced_move)
        for i in range(3):
            x.append(pts[i].x)
            y.append(pts[i].y)
        if not for_fill:
            x.append(pts[0].x)
            y.append(pts[0].y)
        return x, y
    
    def __init__(self, p1, p2, p3, thickness):
        self.points = np.array([p1, p2, p3])
        self.B = np.zeros((3, 6), dtype=current_type)
        self.thickness = thickness

    def get_plane_lines(self):
        lines = []
        for i in range(3):
            p_now = self.points[i]
            p_next = self.points[i if i < 2 else 0]
            line = PlaneLine(p_now.x, p_now.y, p_next.x, p_next.y)
            lines.append(line)
        return lines
    
    def sort_points(self):
        p_new = []
        ys = [p.y for p in self.points]
        y_maxes = np.argwhere(ys == np.amax(ys)).flatten()
        if (len(y_maxes) == 1):
            p_new.append(self.points[y_maxes[0]])
            lst = self.points.tolist()
            lst.pop(y_maxes[0])
            x1 = lst[0].x
            x2 = lst[1].x
            idxs = [0, 1]
            if (x2 < x1):
                idxs = [1, 0]
            for i in idxs:
                p_new.append(lst[i])
        else:
            x1 = self.points[y_maxes[0]].x
            x2 = self.points[y_maxes[1]].x
            idxs = [0, 1]
            if (x2 > x1):
                idxs = [1, 0]
            for i in idxs:
                p_new.append(self.points[y_maxes[i]])
            for i in range(3):
                if i not in y_maxes:
                    p_new.append(self.points[i])
        self.points = np.array(p_new)

    def get_square(self):
        x, y = [], []
        for p in self.points:
            x.append(p.x)
            y.append(p.y)

        x = np.array(x) / 1000
        y = np.array(y) / 1000
    
        C = np.array([
            [1, x[0], y[0]],
            [1, x[1], y[1]],
            [1, x[2], y[2]]
        ], dtype=current_type)

        # print(C)

        return np.linalg.det(C)

    def get_volume(self):
        return self.thickness * self.get_square() / 1000 # толщина

    def get_local_matrix(self, D):
        x, y = [], []
        for p in self.points:
            x.append(p.x)
            y.append(p.y)

        x = np.array(x) / 1000
        y = np.array(y) / 1000
    
        C = np.array([
            [1, x[0], y[0]],
            [1, x[1], y[1]],
            [1, x[2], y[2]]
        ], dtype=current_type)

        IC = np.linalg.inv(C)

        for i in range(3):
            self.B[0][2 * i + 0] = IC[1][i]
            self.B[0][2 * i + 1] = 0.0
            self.B[1][2 * i + 0] = 0.0
            self.B[1][2 * i + 1] = IC[2][i]
            self.B[2][2 * i + 0] = IC[2][i]
            self.B[2][2 * i + 1] = IC[1][i]

        K = np.matmul(np.matmul(np.transpose(self.B), D), self.B, dtype=current_type) * np.abs(np.linalg.det(C) * self.thickness, dtype=current_type) / 2
        return K
    
    def set_strain(self, D, displaysments, move=False):
        delta = np.zeros((6, 1))
        #print(displaysments)
        for i in range(3):
            idx = self.points[i].idx
            delta[i * 2 + 0] = displaysments[idx * 2 + 0]
            delta[i * 2 + 1] = displaysments[idx * 2 + 1]

        sigma_matr = np.dot(np.dot(D, self.B), delta)
        sigma = np.sqrt(sigma_matr[0] ** 2 - sigma_matr[0] * sigma_matr[1] + sigma_matr[1] ** 2 + 3 * sigma_matr[2] ** 2)
        if not move:
            self.sigma_mises = sigma[0]
        else:
            self.sigma_mises_move = sigma[0]

    def move_points(self, displaysments, move=False):
        disps = []
        for p in self.points:
            idx = p.idx
            new_p = Point(p.x + displaysments[idx * 2 + 0], p.y + displaysments[idx * 2 + 1])
            disps.append(new_p)
        if not move:
            self.displaced = disps
        else:
            self.displaced_move = disps

    def get_abs_deformation(self, move=False):
        sum = 0
        for i in range(3):
            sum += get_distance_from_points(self.points[i], self.displaced[i] if not move else self.displaced_move[i])
        return sum
    
    def get_avg_deformation(self, move=False):
        return self.get_abs_deformation(move) / 3

    def __str__(self) -> str:
        return ' '.join([str(p) for p in self.points])

def get_elements_with_diff_nodes(fx_in, fy_in, fx_out, fy_out, t_in_count, thickness):
    t_in = np.linspace(0, 1, t_in_count + 2)
    t_out = np.linspace(0, 1, t_in_count + 3)
    x_in, y_in = np.round(fx_in(t_in), 5), np.round(fy_in(t_in), 5)
    #print(x_in, y_in)
    x_out, y_out = np.round(fx_out(t_out), 5), np.round(fy_out(t_out), 5)
    #print(x_out, y_out)
    els = []
    idx_in = 0
    idx_out = 0
    max_count = t_in_count + t_in_count + 3
    #print(x_in, x_out)
    for i in range(max_count):
        #print(f'i = {i}, i_in = {idx_in}, i_out = {idx_out}')
        p1 = Point(x_in[idx_in], y_in[idx_in])
        p2 = Point(x_out[idx_out], y_out[idx_out])
        p3 = None
        if i % 2 == 0:
            p3 = Point(x_out[idx_out + 1], y_out[idx_out + 1])
            idx_out += 1
        else:
            p3 = Point(x_in[idx_in + 1], y_in[idx_in + 1])
            idx_in += 1
        els.append(Element(p1, p2, p3, thickness))
    return np.array(els)

def get_elements_with_same_nodes(fx_in, fy_in, fx_out, fy_out, t_in_count, thickness):
    t_in = np.linspace(0, 1, t_in_count + 2)
    t_out = np.linspace(0, 1, t_in_count + 2)
    x_in, y_in = np.round(fx_in(t_in), 5), np.round(fy_in(t_in), 5)
    #print(x_in, y_in)
    x_out, y_out = np.round(fx_out(t_out), 5), np.round(fy_out(t_out), 5)
    #print(x_out, y_out)
    els = []
    idx_in = 0
    idx_out = 0
    max_count = t_in_count + t_in_count + 2
    #print(x_in, x_out)
    for i in range(max_count):
        #print(f'i = {i}, i_in = {idx_in}, i_out = {idx_out}')
        p1 = Point(x_in[idx_in], y_in[idx_in])
        p2 = None
        p3 = None
        if i % 2 == 0:
            p2 = Point(x_out[idx_out], y_out[idx_out])
            p3 = Point(x_in[idx_in + 1], y_in[idx_in + 1])
            idx_in += 1
        else:
            p2 = Point(x_out[idx_out], y_out[idx_out])
            p3 = Point(x_out[idx_out + 1], y_out[idx_out + 1])
            idx_out += 1
        els.append(Element(p1, p2, p3, thickness))
    return np.array(els)

def get_elements_by_two_circles(x_in, y_in, x_out, y_out, t_in_count, same, thickness):
    n = len(x_in)
    elements = []
    for i in range(n - 1):
        fx_in, fy_in = get_line_f(x_in[i:i+2], y_in[i:i+2])
        fx_out, fy_out = get_line_f(x_out[i:i+2], y_out[i:i+2])
        els = None
        if same:
            els = get_elements_with_same_nodes(fx_in, fy_in, fx_out, fy_out, t_in_count, thickness)
        else:
            els = get_elements_with_diff_nodes(fx_in, fy_in, fx_out, fy_out, t_in_count, thickness)
        for el in els:
            elements.append(el)
    return elements

def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class Info(object):  
    def __init__(self, d1 = None, 
                 d2 = None, d3 = None, 
                 d4 = None, h1 = None, 
                 h2 = None, needle_count = None, 
                 t_in = None, t_needle = None, 
                 t_out = None, density = None,
                 youngModulus = None, poissonRatio = None):
        if not d1: d1 = 381
        if not d2: d2 = 307
        if not d3: d3 = 127
        if not d4: d4 = 71
        if not h1: h1 = 80
        if not h2: h2 = 60
        if not needle_count: needle_count = 6
        if not t_in: t_in = 152.4
        if not t_out: t_out = 152.4
        if not t_needle: t_needle = 152.4
        if not density: density = 7.900
        if not youngModulus: youngModulus = 2000000
        if not poissonRatio: poissonRatio = 0.25

        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.h1 = h1
        self.h2 = h2
        self.needle_count = needle_count
        self.t_in = t_in
        self.t_out = t_out
        self.t_needle = t_needle
        self.density = density
        self.poissonRatio = poissonRatio
        self.youngModulus = youngModulus

    def __str__(self) -> str:
        return f"d1 = {self.d1}\nd2 = {self.d2}\nd3 = {self.d3}\nd4 = {self.d4}\nh1 = {self.h1}\nh2 = {self.h2}"

class Circle(object):
    def __init__(self, x, y, r, t):
        self.x = x
        self.y = y
        self.r = r
        self.t = t

    def __len__(self):
        return len(self.x)
        
class Needles(object):
    def __init__(self, x1, y1, x2, y2) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def devide(self):
        lines = []
        for i in range(len(self.x1)):
            if (i + 1) % 3 != 2:
                x = [self.x1[i], self.x2[i]]
                y = [self.y1[i], self.y2[i]]
                lines.append({'x' : x, 'y' : y})
        return lines
    
    def make_elements(self, step, thickness):
        elements = []
        quads = []
        n = len(self.x1)
        for i in range(0, n, 3):
            line1 = PlaneLine(self.x1[i], self.y1[i], self.x2[i], self.y2[i])
            line2 = PlaneLine(self.x1[i + 1], self.y1[i + 1], self.x2[i + 1], self.y2[i + 1])
            line3 = PlaneLine(self.x1[i + 2], self.y1[i + 2], self.x2[i + 2], self.y2[i + 2])
            quads.append([line1, line2])
            quads.append([line3, line2])
        distance = quads[0][0].get_line_distance()
        count = int(np.floor(distance / step))
        print(f'distance is {distance}, count is {count}')
        for q in quads:
            els = get_elements_with_same_nodes(q[1].fx, q[1].fy, q[0].fx, q[0].fy, count - 3, thickness)
            for e in els:
                elements.append(e)
        return elements

class PlotGenerator(object):
    @staticmethod
    def generate_circles_by_needle_found(r_counted, r_add, h, needle_count):
        alpha = np.arccos(1 - (h ** 2)/(2 * r_counted ** 2))
        step = 2 * np.pi / needle_count
        bias = alpha / 4
        diff = step - 2 * bias 
        count = int(np.floor(diff / bias))
        #print(count)
        t = []
        t_needles = []
        for i in range(needle_count):
            t.append(step * i - bias)
            t.append(step * i)
            t.append(step * i + bias)
            for j in range(count):
                t.append(step * i + bias + diff / count * (j + 1))
            t_needles.append(step * i - bias)
            t_needles.append(step * i)
            t_needles.append(step * i + bias)
        #t.append(-bias)
        t = np.array(t)
        # print(t)
        cx, cy = PlotGenerator.get_circles_coords_by_angles(r_counted, t)
        nx, ny = PlotGenerator.get_circles_coords_by_angles(r_counted, t_needles)
        cx_add, cy_add = PlotGenerator.get_circles_coords_by_angles(r_add, t)
        return cx, cy, cx_add, cy_add, nx, ny, t
    
    @staticmethod
    def get_circles_coords_by_angles(r, t):
        return np.round(r * np.cos(t), 5), np.round(r * np.sin(t), 5)
    
    @staticmethod
    def get_circles_between_circles(c_in, c_out):
        circles = []
        step = np.max([get_distance(c_in.x[i], c_in.y[i], c_in.x[i + 1], c_in.y[i + 1]) for i in range(len(c_in) - 1)])
        circles = []
        step = step * np.cos(np.pi / 6)
        count = (abs(c_out.r - c_in.r) / step)
        if count < 3:
            step = abs(c_out.r - c_in.r) / 2
        r = c_out.r
        while(True):
            x, y = PlotGenerator.get_circles_coords_by_angles(r, c_out.t)
            circles.append(Circle(x, y, r, c_out.t))
            if abs(r - step - c_in.r) < step:
                break
            else:
                r -= step
        circles.append(c_in)
        return circles

    def __init__(self, info : Info = None):
        self.points = []
        self.elements = []
        if not info: info = Info()
        self.info = info

    def generate_circles(self):
        cx3, cy3, cx4, cy4, nin_x, nin_y, t_in = PlotGenerator.generate_circles_by_needle_found(self.info.d3 / 2, 
                                                                                                self.info.d4 / 2, 
                                                                                                self.info.h1, 
                                                                                                self.info.needle_count)
        cx2, cy2, cx1, cy1, nout_x, nout_y, t_out = PlotGenerator.generate_circles_by_needle_found(self.info.d2 / 2, 
                                                                                                   self.info.d1 / 2, 
                                                                                                   self.info.h2, 
                                                                                                   self.info.needle_count)
        self.c1 = Circle(cx1, cy1, self.info.d1 / 2, t_out)
        self.c2 = Circle(cx2, cy2, self.info.d2 / 2, t_out)
        self.c3 = Circle(cx3, cy3, self.info.d3 / 2, t_in)
        self.c4 = Circle(cx4, cy4, self.info.d4 / 2, t_in)
        self.needles = Needles(nin_x, nin_y, nout_x, nout_y)

    def generate_intermeditate_circles(self):
        self.circles_in = PlotGenerator.get_circles_between_circles(self.c4, self.c3)
        self.circles_out = PlotGenerator.get_circles_between_circles(self.c2, self.c1)

    def make_elemenets(self):
        self.elements = []
        t = 0
        cs = [self.circles_in, self.circles_out]
        sizes = []
        for c in cs:
            for i in range(len(c) - 1):
                thickness = self.info.t_in if c in self.circles_in else self.info.t_out
                for e in get_elements_by_two_circles(c[i].x, c[i].y, c[i + 1].x, c[i + 1].y, t, True, thickness):
                    self.elements.append(e)
            e = self.elements[len(self.elements) - 1]
            size_lines = [line.get_line_distance() for line in e.get_plane_lines()]
            size = np.average(size_lines)
            sizes.append(size)
        size = np.average(sizes)
        #print(f'size is {size}')
        els = self.needles.make_elements(size * 2, self.info.t_needle)
        #print(f'els count = {len(els)}')
        for e in els:
            self.elements.append(e)

    def set_points_indices(self):
        i = 0
        while i < len(self.elements):
            if self.elements[i].get_square() == 0:
                self.elements.pop(i)
            else:
                i += 1

        self.points = []
        for e in self.elements:
            for p in e.points:
                if p not in self.points:
                    self.points.append(p) 

        for e in self.elements:
            e.sort_points()

        for e in self.elements:
            for p in e.points:
                idx = self.points.index(p)
                p.idx = idx

        self.n = len(self.points)

    def fix_points(self):
        fixed_points = []
        r = self.info.d4 / 2
        for p in self.points:
            d = get_distance(0, 0, p.x, p.y)
            #print(d)
            if np.abs(r - d) < 1e-3:
                fixed_points.append(p)
        self.fixed = fixed_points

    def attach_forces(self, object_height, f):
        self.forces = np.zeros((self.n * 2, 1))
        p_max = Point(np.Inf, -np.Inf)
        for p in self.points:
            if np.abs(get_distance(0, 0, p.x, p.y) - self.info.d1 / 2 )< 1e-5 and p.x < 0 and p.y > p_max.y and p.y <= -self.info.d1/2 + object_height and p.y <= 0: 
                p_max = p

        print(str(p_max))
        
        self.points_forces = [p_max]
        angle = np.arctan(PlaneLine(0, 0, p_max.x, p_max.y).k)
        for p in self.points:
            p_angle = np.arctan(PlaneLine(0, 0, p.x, p.y).k)
            if np.abs(get_distance(0, 0, p.x, p.y) - self.info.d1 / 2) < 1e-5 and abs(angle - p_angle) <= np.pi / 18 and p.x < 0:
                idx = p.idx
                self.forces[idx * 2 + 0] = f * np.cos(p_angle)
                self.forces[idx * 2 + 1] = f * np.sin(p_angle)
                self.points_forces.append(p)

    def attach_move_forces(self, speed, _in=True):
        speed = speed * 1000 / 3600 # м/с
        self.move_forces = np.zeros((self.n * 2, 1))
        #print(self.move_forces.shape)
        for e in self.elements:
            weight = e.get_volume() * self.info.density * 1000 # кг
            for p in e.points:
                if p not in self.fixed:
                    r = get_distance(0, 0, p.x, p.y) / 1000
                    a = speed ** 2 / r # м/с^2
                    f = weight * a # Н
                    angel = np.arctan(np.true_divide(p.y, p.x))
                    f_x = f * np.cos(angel)
                    f_y = f * np.sin(angel)
                    if not _in:
                        f_x = -f_x
                        f_y = -f_y
                    idx = p.idx
                    self.move_forces[idx * 2 + 0] += f_x
                    self.move_forces[idx * 2 + 1] += f_y                
            
    def set_displacements(self, displacements, move=False):
        for e in self.elements:
            e.move_points(displacements, move)

    def get_plot_info(self):
        x, y = [], []
        cs = [self.c1, self.c2, self.c3, self.c4]
        for c in cs:
            x.append(c.x)
            y.append(c.y)
        for l in self.needles.devide():
            x.append(l['x'])
            y.append(l['y'])
        return x, y

    def get_all_plot_info(self):
        x, y = [], []
        for c in self.circles_in:
            x.append(c.x)
            y.append(c.y)
        for c in self.circles_out:
            x.append(c.x)
            y.append(c.y)
        for l in self.needles.devide():
            x.append(l['x'])
            y.append(l['y'])
        return x, y
    
class Calculator(object):
    def __init__(self, generator : PlotGenerator):
        self.gen = generator
        self.gens = []
        self.u = self.gen.info.poissonRatio
        self.E = self.gen.info.youngModulus
        self.D = np.array([[1, self.u, 0],
                           [self.u, 1, 0],
                           [0, 0, (1 - self.u) / 2]], dtype=current_type)
        self.D = self.D * self.E / (1  - self.u ** 2)

    def set_global_matrix(self):
        n = self.gen.n
        self.K = np.zeros((n * 2, n * 2))
        count = 0
        for e in self.gen.elements:
            try:
                k_local = e.get_local_matrix(self.D)
                for i in range(3):
                    for j in range(3):
                        i_k = 2 * e.points[i].idx
                        j_k = 2 * e.points[j].idx
                        self.K[i_k + 0][j_k + 0] += k_local[2 * i + 0][2 * j + 0]
                        self.K[i_k + 0][j_k + 1] += k_local[2 * i + 0][2 * j + 1]
                        self.K[i_k + 1][j_k + 0] += k_local[2 * i + 1][2 * j + 0]
                        self.K[i_k + 1][j_k + 1] += k_local[2 * i + 1][2 * j + 1]
            except:
                count += 1
            
        for p in self.gen.fixed:
            pos_point = p.idx * 2
            size = len(self.K)
            for i in range(size):
                # if x:
                self.K[i][pos_point] = 0
                self.K[pos_point][i] = 0
                # if y:
                self.K[i][pos_point + 1] = 0
                self.K[pos_point + 1][i] = 0

            #if x:
            self.K[pos_point][pos_point] = 1
            #if y:
            self.K[pos_point + 1][pos_point + 1] = 1
    
    def calculate_displacements(self, move=False):
        F = self.gen.forces if not move else self.gen.move_forces
        disps = (np.linalg.solve(self.K, F) * 1000).flatten()
        #disps = np.linalg.lstsq(self.K, F)[0] * 1000
        #disps = scipy.linalg.solve(self.K, F) * 1000
        if not move:
            self.displacements = disps
        else:
            self.displacements_move = disps
        return disps

    def calculate_strain(self, move=False):
        strains = []
        for e in self.gen.elements:
            if not move:
                e.set_strain(self.D, self.displacements, move)
                strains.append(e.sigma_mises)
            else:
                e.set_strain(self.D, self.displacements_move, move)
                strains.append(e.sigma_mises_move)