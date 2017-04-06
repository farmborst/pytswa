# -*- coding: utf-8 -*-
''' accpy.gui.layout
author:     felix.kramer(at)physik.hu-berlin.de
'''
try:
    from Tkinter import (Label, StringVar, IntVar, DoubleVar, BooleanVar,
                         Entry, Button, OptionMenu, Checkbutton)
    from ttk import (Notebook, Frame)
except:
    from tkinter import (Label, StringVar, IntVar, DoubleVar, BooleanVar,
                         Entry, Button, OptionMenu, Checkbutton)
    from tkinter.ttk import (Notebook, Frame)


def packwidget(button, side, fill, expand):
    button.pack(side=side, fill=fill, expand=expand)


# create and name tabs in root
def cs_tabbar(root, w, h, names):
    nb = Notebook(root, width=w, height=h)
    tabs = [Frame(nb) for i in range(len(names))]  # 5 tabs
    [nb.add(tabs[i], text=name) for i, name in enumerate(names)]
    nb.pack()
    return tabs


# create and set tkinter stringvar
def cs_str(name):
    svar = StringVar()
    svar.set(name)
    return svar


def cs_bln(val):
    bvar = BooleanVar()
    bvar.set(val)
    return bvar


# create and set tkinter IntVar
def cs_int(value):
    ivar = IntVar()
    ivar.set(int(value))
    return ivar


# create and set tkinter DoubleVar
def cs_dbl(value):
    dvar = DoubleVar()
    dvar.set(float(value))
    return dvar


# create, position and set tkinter label
def cs_label(root, r, c, name, label_conf={}, grid_conf={}):
    labelstr = cs_str(name)
    label = Label(master=root, textvariable=labelstr, **label_conf)
    label.grid(row=r, column=c, **grid_conf)
    return labelstr, label


# create, position and set IntVar label
def cs_Intentry(root, r, c, value, **kwargs):
    entryint = cs_int(value)
    entry = Entry(root, textvariable=entryint)
    entry.grid(row=r, column=c, **kwargs)
    return entry


# create, position and set DoubleVar label
def cs_Dblentry(root, r, c, value, **kwargs):
    entryint = cs_dbl(value)
    entry = Entry(root, textvariable=entryint)
    entry.grid(row=r, column=c, **kwargs)
    return entry


# create, position and set StringVar label
def cs_Strentry(root, r, c, value, **kwargs):
    entrystr = cs_str(value)
    entry = Entry(root, textvariable=entrystr)
    entry.grid(row=r, column=c, **kwargs)
    return entry


# create, position and set button
def cs_button(root, r, c, label, action, **kwargs):
    button = Button(master=root, text=label, command=action)
    button.grid(row=r, column=c, **kwargs)
    return button


# create and pack button
def cp_button(root, label, action, side="top", fill="both", expand=True):
    button = Button(master=root, text=label, command=action)
    packwidget(button, side, fill, expand)
    return button


# create, position and set button
def cs_checkbox(root, r, c, label, boolval, **kwargs):
    entrybl = cs_bln(boolval)
    checkbox = Checkbutton(master=root, text=label, onvalue=True, offvalue=False, variable=entrybl)
    checkbox.grid(row=r, column=c, **kwargs)
    return entrybl


def cs_dropd(root, r, c, options, action=None, **kwargs):
    startvalue = cs_str('')  # cs_str(options[0])
    dropdown = OptionMenu(root, startvalue, *options)
    dropdown.grid(row=r, column=c, **kwargs)
    if action is not None:
        startvalue.trace('w', action)
    return startvalue, dropdown
