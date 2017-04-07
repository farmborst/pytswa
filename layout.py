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


# create and name tabs in root
def cs_tabbar(root, w, h, names):
    nb = Notebook(root, width=w, height=h)
    tabs = [Frame(nb) for i in range(len(names))]  # 5 tabs
    [nb.add(tabs[i], text=name) for i, name in enumerate(names)]
    nb.pack()
    return tabs


# create, position and set tkinter label
def cs_label(root, r, c, name, label_conf={}, grid_conf={}):
    labelstr = StringVar()
    labelstr.set(name)
    label = Label(master=root, textvariable=labelstr, **label_conf)
    label.grid(row=r, column=c, **grid_conf)
    return labelstr, label


# create, position and set IntVar label
def cs_Intentry(root, r, c, value, entry_conf={}, grid_conf={}):
    entryint = IntVar()
    entryint.set(int(value))
    entry = Entry(root, textvariable=entryint, **entry_conf)
    entry.grid(row=r, column=c, **grid_conf)
    return entry


# create, position and set DoubleVar label
def cs_Dblentry(root, r, c, value, entry_conf={}, grid_conf={}):
    entrydbl = DoubleVar()
    entrydbl.set(float(value))
    entry = Entry(root, textvariable=entrydbl, **entry_conf)
    entry.grid(row=r, column=c, **grid_conf)
    return entry


# create, position and set StringVar label
def cs_Strentry(root, r, c, value, entry_conf={}, grid_conf={}):
    entrystr = StringVar()
    entrystr.set(value)
    entry = Entry(root, textvariable=entrystr, **entry_conf)
    entry.grid(row=r, column=c, **grid_conf)
    return entry


# create, position and set button
def cs_button(root, r, c, label, action, button_conf={}, grid_conf={}):
    button = Button(master=root, text=label, command=action, **button_conf)
    button.grid(row=r, column=c, **grid_conf)
    return button


# create and pack button
def cp_button(root, label, action, button_conf={}, pack_conf={'side':"top", 'fill':"both", 'expand':'True'}):
    button = Button(master=root, text=label, command=action, **button_conf)
    button.pack(**pack_conf)
    return button


# create, position and set button
def cs_checkbox(root, r, c, label, boolval, box_conf={}, grid_conf={}):
    entrybl = BooleanVar()
    entrybl.set(boolval)
    checkbox = Checkbutton(master=root, text=label, onvalue=True, offvalue=False, variable=entrybl, **box_conf)
    checkbox.grid(row=r, column=c, **grid_conf)
    return entrybl


def cs_dropd(root, r, c, options, action=None, drop_conf={}, grid_conf={}):
    startvalue = StringVar()
    startvalue.set('')  # options[0]
    dropdown = OptionMenu(root, startvalue, **drop_conf)
    dropdown.grid(row=r, column=c, **grid_conf)
    if action is not None:
        startvalue.trace('w', action)
    return startvalue, dropdown
