from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from gui import *
from geometry import *
import matplotlib.pyplot as plt
from ctypes import windll
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
info = Info()
generator = None
calculator = None

def set_net(control1 : Widget, control2 : Widget, frame : Frame, canvas : Canvas):
    d1 = float(control1.children['textBoxD1'].get())
    d2 = float(control1.children['textBoxD2'].get())
    d3 = float(control1.children['textBoxD3'].get())
    d4 = float(control1.children['textBoxD4'].get())
    h1 = float(control1.children['textBoxH1'].get())
    h2 = float(control1.children['textBoxH2'].get())

    E = float(control2.children['textBoxE'].get())
    u = float(control2.children['textBoxu'].get())

    info.d1 = d1
    info.d2 = d2
    info.d3 = d3
    info.d4 = d4
    info.h1 = h1
    info.h2 = h2
    info.youngModulus = E
    info.poissonRatio = u

    global generator 
    generator = PlotGenerator(info)
    generator.generate_circles()
    generator.generate_intermeditate_circles()
    generator.make_elemenets()
    generator.set_points_indices()
    generator.fix_points()
    figure = plt.figure(figsize=(4, 6))
    ax = figure.add_subplot(111)
    
    for widgets in frame.winfo_children():
        widgets.destroy()
    
    for e in generator.elements:
        x, y = Element.get_x_y(e)
        ax.plot(x, y, color='black')

    canvas = FigureCanvasTkAgg(figure, frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def calculate_coll(control : Widget, frame : Frame, canvas : Canvas):
    global generator
    global calculator
    if not calculator: 
        calculator = Calculator(generator)
        calculator.set_global_matrix()

    H = float(control.children['textBoxH'].get())
    F = float(control.children['textBoxF'].get())
    
    generator.attach_forces(H, F)
    disps = calculator.calculate_displacements(move=False)
    generator.set_displacements(disps, move=False)
    calculator.calculate_strain(move=False)
    figure = plt.figure(figsize=(8, 4))
    ax_deformation = figure.add_subplot(121)
    ax_strain = figure.add_subplot(122)
    
    for widgets in frame.winfo_children():
        widgets.destroy()

    ax_deformation.set_title('Деформация')
    plot_deformation(ax_deformation, generator.elements, False)
    ax_strain.set_title('Напряжение')
    plot_strain(ax_strain, generator.elements, False)

    canvas = FigureCanvasTkAgg(figure, frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=0)

def calculate_move(control : Widget, frame : Frame, canvas : Canvas):
    global generator
    global calculator
    if not calculator: 
        calculator = Calculator(generator)
        calculator.set_global_matrix()

    V = float(control.children['textBoxV'].get())
    generator.attach_move_forces(V)
    disps = calculator.calculate_displacements(move=True)
    generator.set_displacements(disps, move=True)
    calculator.calculate_strain(move=True)

    figure = plt.figure(figsize=(8, 4))
    ax_deformation = figure.add_subplot(121)
    ax_strain = figure.add_subplot(122)
    
    for widgets in frame.winfo_children():
        widgets.destroy()

    ax_deformation.set_title('Деформация')
    plot_deformation(ax_deformation, generator.elements, True)
    ax_strain.set_title('Напряжение')
    plot_strain(ax_strain, generator.elements, True)

    canvas = FigureCanvasTkAgg(figure, frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=0)

def plot_deformation(ax, elements, move : bool):
    max = 0
    min = np.Inf
    min = np.Inf
    for e in elements:
        deformation = e.get_avg_deformation(move=move)
        if deformation > max:
            max = deformation
        elif deformation < min:
            min = deformation
    max_diff = np.abs(max - min)
    for e in elements:
        diff = np.abs(e.get_avg_deformation(move=move) - min)
        coeff = diff / max_diff
        color = (coeff, 0, 1 - coeff, 1)
        x, y = Element.get_x_y(e, old_points=False, for_fill=True, move=move)
        ax.fill(x, y, color=color)
        x, y = Element.get_x_y(e, old_points=False, move=move)
        ax.plot(x, y, color='black')

def plot_strain(ax, elements, move : bool):
    max = 0
    min = np.Inf
    for e in elements:
        if move:
            if e.sigma_mises_move > max:
                max = e.sigma_mises_move
            elif e.sigma_mises_move < min:
                min = e.sigma_mises_move
        else:
            if e.sigma_mises > max:
                max = e.sigma_mises
            elif e.sigma_mises < min:
                min = e.sigma_mises
    max_diff = np.abs(max - min)
    for e in elements:
        diff = (e.sigma_mises_move if move else e.sigma_mises) - min
        coeff = diff / max_diff
        # print(coeff)
        color = (coeff, 0, 1 - coeff, 1)
        x, y = Element.get_x_y(e, old_points=False, for_fill=move, move=move)
        ax.fill(x, y, color=color)
        x, y = Element.get_x_y(e, old_points=False, move=move)
        ax.plot(x, y, color='black')

if __name__=='__main__':
    windll.shcore.SetProcessDpiAwareness(1)
    root = Tk()
    root.title('Курсовая работа Гузов Е.А.')
    root.geometry('1500x800')
    root.resizable(False, False)
    root.configure(background=SMPL_BG)

    for c in range(4): root.columnconfigure(index=c, weight=1)
    for r in range(1): root.rowconfigure(index=r, weight=1)

    setting_plot(root)
    setting_control(root, set_net, calculate_coll, calculate_move)

    root.mainloop()