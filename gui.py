from tkinter import *
from tkinter import ttk
from tkinter import messagebox

canvaces = []
canvasframes = []

from static_gui import *
def setting_plot(window : Tk):
    plotframe = Frame(window)
    plotframe.grid(row = 0, column = 0, columnspan=3, padx=5)
    labelname = Label(plotframe, text='Сетка конечных элементов', font=FONT_HEADER, background=SMPL_BG)
    labelname.pack()

    setting_tab_control(plotframe)

def setting_tab_control(plotframe):
    style = ttk.Style()
    style.theme_use('default')
    style.configure('TNotebook.Tab', font=FONT_INPUT)

    tabcontrol = ttk.Notebook(plotframe, width=1000, height=500)
    tab1 = Frame(tabcontrol)
    tab1.configure(background=SMPL_BG)
    canvasframes.append(tab1)
    canvas1 = Canvas(tab1)
    canvas1.pack()
    canvaces.append(canvas1)
    tab2 = Frame(tabcontrol)
    canvas2 = Canvas(tab2)
    canvas2.pack()
    canvaces.append(canvas2)
    canvasframes.append(tab2)
    tab2.configure(background=SMPL_BG)
    tab3 = Frame(tabcontrol)
    canvas3 = Canvas(tab3)
    canvas3.pack()
    canvaces.append(canvas3)
    canvasframes.append(tab3)
    tab3.configure(background=SMPL_BG)
    tabcontrol.add(tab1, text='Сетка')
    tabcontrol.add(tab2, text='Деформация при наезде')
    tabcontrol.add(tab3, text='Состояние в движении')
    tabcontrol.pack(ipadx=0)

def setting_control(window : Tk, net_func, coll_func, move_func):
    controlframe = Frame(window, background=SMPL_BG, pady=10)
    controlframe.grid(row = 0, column = 3, columnspan=1, padx=10)
    
    valuesframe = setting_detail_vars(controlframe)
    materialframe = setting_material_var(controlframe)
    setting_net_button(controlframe, net_func, valuesframe, materialframe)
    collframe = setting_collision_var(controlframe)
    mvframe = setting_movement_var(controlframe)

    setting_collision_button(controlframe, coll_func, collframe)
    setting_move_button(controlframe, move_func, mvframe)

def setting_detail_vars(controlframe):
    labelname = Label(controlframe, text='Настройки', font=FONT_HEADER, background=SMPL_BG)
    labelname.pack(pady=HEADER_PADY, padx=HEADER_PADX)
    labelsizes = Label(controlframe, text='Размеры', font=FONT_TEXT, background=SMPL_BG)
    labelsizes.pack()
    valuesframe = Frame(controlframe, background=SMPL_BG)
    textVars = ['381', '307', '127', '71', '80', '60', '152.4', '152.4', '152.4']
    labelD1 = Label(valuesframe, text='D1, мм', font=FONT_INPUT, background=SMPL_BG)
    labelD1.grid(row=0, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxD1 = Entry(valuesframe, name='textBoxD1', font=FONT_INPUT)
    textBoxD1.insert(0, textVars[0])
    textBoxD1.grid(row=0, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelD2 = Label(valuesframe, text='D2, мм', font=FONT_INPUT, background=SMPL_BG)
    labelD2.grid(row=1, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxD2 = Entry(valuesframe, name='textBoxD2', font=FONT_INPUT)
    textBoxD2.insert(0, textVars[1])
    textBoxD2.grid(row=1, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelD3 = Label(valuesframe, text='D3, мм', font=FONT_INPUT, background=SMPL_BG)
    labelD3.grid(row=2, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxD3 = Entry(valuesframe, name='textBoxD3', font=FONT_INPUT)
    textBoxD3.insert(0, textVars[2])
    textBoxD3.grid(row=2, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelD4 = Label(valuesframe, text='D4, мм', font=FONT_INPUT, background=SMPL_BG)
    labelD4.grid(row=3, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxD4 = Entry(valuesframe, name='textBoxD4', font=FONT_INPUT)
    textBoxD4.insert(0, textVars[3])
    textBoxD4.grid(row=3, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelH1 = Label(valuesframe, text='H1, мм', font=FONT_INPUT, background=SMPL_BG)
    labelH1.grid(row=4, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxH1 = Entry(valuesframe, name='textBoxH1', font=FONT_INPUT)
    textBoxH1.insert(0, textVars[4])
    textBoxH1.grid(row=4, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelH2 = Label(valuesframe, text='H2, мм', font=FONT_INPUT, background=SMPL_BG)
    labelH2.grid(row=5, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxH2 = Entry(valuesframe, name='textBoxH2', font=FONT_INPUT)
    textBoxH2.insert(0, textVars[5])
    textBoxH2.grid(row=5, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelIn = Label(valuesframe, text='Толщина внутри, мм', font=FONT_INPUT, background=SMPL_BG)
    labelIn.grid(row=6, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxIn = Entry(valuesframe, name='textBoxIn', font=FONT_INPUT)
    textBoxIn.insert(0, textVars[6])
    textBoxIn.grid(row=6, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelNeedle = Label(valuesframe, text='Толщина спиц, мм', font=FONT_INPUT, background=SMPL_BG)
    labelNeedle.grid(row=7, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxNeedle = Entry(valuesframe, name='textBoxNeedle', font=FONT_INPUT)
    textBoxNeedle.insert(0, textVars[7])
    textBoxNeedle.grid(row=7, column=1, padx=INPUT_PADX, pady=INPUT_PADY)
    
    labelOut = Label(valuesframe, text='Толщина снаружи, мм', font=FONT_INPUT, background=SMPL_BG)
    labelOut.grid(row=8, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxOut = Entry(valuesframe, name='textBoxOut', font=FONT_INPUT)
    textBoxOut.insert(0, textVars[8])
    textBoxOut.grid(row=8, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    valuesframe.pack()
    return valuesframe

def setting_material_var(controlframe):
    charslabel = Label(controlframe, text='Характеристики материала', font=FONT_TEXT, background=SMPL_BG)
    charslabel.pack(pady=HEADER_PADY, padx=HEADER_PADX)
    charsframe = Frame(controlframe, background=SMPL_BG)
    textVars = ['2000000', '0.25']
    labelE = Label(charsframe, text='Модуль Юнга, МПа', font=FONT_INPUT, background=SMPL_BG)
    labelE.grid(row=0, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxE = Entry(charsframe, name='textBoxE', font=FONT_INPUT)
    textBoxE.insert(0, textVars[0])
    textBoxE.grid(row=0, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelu = Label(charsframe, text='Коэффициент Пуассона', font=FONT_INPUT, background=SMPL_BG)
    labelu.grid(row=1, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxu = Entry(charsframe, name='textBoxu', font=FONT_INPUT)
    textBoxu.insert(0, textVars[1])
    textBoxu.grid(row=1, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    charsframe.pack()
    return charsframe

def setting_collision_var(controlframe):
    collisionlabel = Label(controlframe, text='Наезд на препятствие', font=FONT_TEXT, background=SMPL_BG)
    collisionlabel.pack(pady=HEADER_PADY, padx=HEADER_PADX)
    collframe = Frame(controlframe, background=SMPL_BG)
    textVars = ['50', '100']

    labelH = Label(collframe, text='Высота объекта, мм', font=FONT_INPUT, background=SMPL_BG)
    labelH.grid(row=0, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxH = Entry(collframe, name='textBoxH', font=FONT_INPUT)
    textBoxH.insert(0, textVars[0])
    textBoxH.grid(row=0, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    labelF = Label(collframe, text='Сила, Н', font=FONT_INPUT, background=SMPL_BG)
    labelF.grid(row=1, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxF = Entry(collframe, name='textBoxF', font=FONT_INPUT)
    textBoxF.insert(0, textVars[0])
    textBoxF.grid(row=1, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    collframe.pack()
    return collframe

def setting_movement_var(controlframe):
    movementlabel = Label(controlframe, text='В движении', font=FONT_TEXT, background=SMPL_BG)
    movementlabel.pack(pady=HEADER_PADY, padx=HEADER_PADX)
    mvframe = Frame(controlframe, background=SMPL_BG)
    var = ['100']
    labelV = Label(mvframe, text='Скорость движения, км/ч', font=FONT_INPUT, background=SMPL_BG)
    labelV.grid(row=0, column=0, padx=INPUT_PADX, pady=INPUT_PADY)
    textBoxV = Entry(mvframe, name='textBoxV', font=FONT_INPUT)
    textBoxV.insert(0, var[0])
    textBoxV.grid(row=0, column=1, padx=INPUT_PADX, pady=INPUT_PADY)

    mvframe.pack()
    return mvframe

def setting_net_button(controlframe, net_func, frame1, frame2):
    butt = Button(controlframe, text='Построить сетку', font=FONT_BUTTON, width=35, command=(lambda : net_func(frame1, frame2, canvasframes[0], canvaces[0])))
    butt.pack(padx=BUTTON_PADX, pady=BUTTON_PADY)

def setting_collision_button(controlframe, coll_func, frame):
    butt = Button(controlframe, text='Расчитать деформацию при наезде', font=FONT_BUTTON, width=35, command=(lambda : coll_func(frame, canvasframes[1], canvaces[1])))
    butt.pack(padx=BUTTON_PADX, pady=BUTTON_PADY)

def  setting_move_button(controlframe, move_func, frame):
    butt = Button(controlframe, text='Состояние в движении', font=FONT_BUTTON, width=35, command=(lambda : move_func(frame, canvasframes[2], canvaces[2])))
    butt.pack(padx=BUTTON_PADX, pady=BUTTON_PADY)