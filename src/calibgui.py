import os
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk
from tkinter import Frame
from tkinter import filedialog as fd

from src.utils import saved_normalize, saved_to_info, get_precal_coords, get_calibrated_points
from src.calib import calib_2cam, calib_ncam, calib_1cam

img_filetypes = (
    ('PNG files', '*.png'),
    ('JPG files', '*.jpg'),
    ('JPEG files', '*.jpeg'),
    ('All files', '*.*')
)

npy_filetypes = (
    ('NPY Files', '*.npy'),
    ('All files', '*.*')
)


def load_image(file_path, scale, resize=True):
    img1 = Image.open(file_path)
    w, h = img1.size
    target_w = int(w // scale)
    target_h = int(h // scale)
    if resize:
        img1 = img1.resize(
            (target_w, target_h), Image.Resampling.LANCZOS)
    img1 = ImageTk.PhotoImage(img1)
    return img1


class CalibGUI(tk.Frame):
    def __init__(self, master=None, args={}):
        Frame.__init__(self, master)
        self.master = master
        self.args = args

        main_window_width = 250
        main_window_height = 900
        master.geometry('{}x{}'.format(main_window_width, main_window_height))

        self.canvas_width = 2000
        self.canvas_height = 2000

        ###### Select image files ######
        # State Vars
        self.img_filepaths = []
        self.img_files = []
        self.minimap_path = ''
        self.minimap_filename = ''
        self.minimap = None
        self.idx_img1 = 0
        self.idx_img2 = 1
        self.idx_minimap = 2
        self.point_idx = 0
        self.point_lbl = 0
        self.floor_points = []
        self.save_dir = None
        self.point_data = {}
        self.calibrated = []
        self.calib_param = {}
        
        # Set a counter for the rows
        curr_row = 0

        # TODO: Fix logo
        # Create an object of tkinter ImageTk
        # img = Image.open("./asset/logo.png")
        # img = img.resize((111, 90), Image.Resampling.LANCZOS)
        # self.logo_img = ImageTk.PhotoImage(img)

        # # Create a photoimage object of the image in the path
        # logo_label = tk.Label(master, image=self.logo_img)
        # logo_label.grid(
        #     column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=0, pady=0)
        curr_row += 1

        # Button Select images
        self.button_images = ttk.Button(
            master,
            text='Images',
            command=self.select_img_files
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)

        # Button Select minimap
        self.button_minimap = ttk.Button(
            master,
            text='Minimap',
            command=self.select_minimap
        ).grid(
            column=1, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)
        curr_row += 1

        # Button load points
        self.button_images = ttk.Button(
            master,
            text='Points',
            command=self.load_point_data
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)
        
        # Button load parameters
        self.button_images = ttk.Button(
            master,
            text='Parameters',
            command=self.load_parameters
        ).grid(
            column=1, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)
        curr_row += 1

        # Window 1 Text box
        tk.Label(master, text="Image 1").grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=0)
        curr_row += 1

        # File List boxes
        self.file_list_img1 = tk.Listbox(
            master, height=5, width=20, selectmode='extended')
        self.file_list_img1.grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        # Window 2 Text box
        tk.Label(master, text="Image 2").grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=0)
        curr_row += 1

        self.file_list_img2 = tk.Listbox(
            master, height=5, width=20, selectmode='extended')
        self.file_list_img2.grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        # Point list text box
        tk.Label(master, text="Points").grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=0)
        curr_row += 1

        self.previous_l1_selections = None
        self.previous_l2_selections = None
        self.previous_point_selections = None

        # Point list
        self.point_list = tk.Listbox(
            master, 
            height=5, width=20, 
            selectforeground='red', selectmode='extended', activestyle='none')
        self.point_list.grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        # Image & minimap scaling option menu
        tk.Label(master, text="Scale").grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=0)
        curr_row += 1

        img_scale_opt = ['1.0', '2.0', '4.0']
        self.img_scale = tk.StringVar()
        self.img_scale.set(img_scale_opt[0])
        self.previous_img_scale = float(self.img_scale.get())

        self.img_scaling_option_menu = tk.OptionMenu(
            master, self.img_scale, *img_scale_opt)
        self.img_scaling_option_menu.grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)
        curr_row += 1

        # Calibrate Buttons
        self.button_calibrate_2cams = ttk.Button(
            master,
            text='Calibrate 2 Cameras',
            command=self.calibrate_2cams
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        # Calibrate Buttons
        self.button_calibrate_ncams = ttk.Button(
            master,
            text='Calibrate N Cameras',
            command=self.calibrate_ncams
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        # Calibrate Buttons
        self.button_recalibrate = ttk.Button(
            master,
            text='Add 1 Camera',
            command=self.calibrate_1cam
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        self.button_eval_calibration = ttk.Button(
            master,
            text='Evaluate Calibration',
            command=self.display_image_points_to_minimap_points
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, columnspan=2, padx=10, pady=10)
        curr_row += 1

        self.button_load_state = ttk.Button(
            master,
            text='Load',
            command=self.load_state
        ).grid(
            column=0, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)
        
        self.button_save_state = ttk.Button(
            master,
            text='Save',
            command=self.save_state
        ).grid(
            column=1, row=curr_row, sticky=tk.W+tk.E, padx=10, pady=10)
        
        curr_row += 1

        ###### Image Display #######
        canvas_size = '{}x{}'.format(200, 200)
        # Display Canvas in new window
        window_img1 = tk.Toplevel(master)
        window_img1.title("Window 1")
        window_img1.geometry(canvas_size)
        self.canvas_img1 = tk.Canvas(
            window_img1, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_img1.pack()

        # Display Canvas in new window
        window_img2 = tk.Toplevel(master)
        window_img2.title("Window 2")
        window_img2.geometry(canvas_size)
        self.canvas_img2 = tk.Canvas(
            window_img2, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_img2.pack()

        # Display Canvas in new window
        window_img3 = tk.Toplevel(master)
        window_img3.title("Minimap")
        window_img3.geometry(canvas_size)
        self.canvas_minimap = tk.Canvas(
            window_img3, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas_minimap.pack()

        # Key bindings for image navigation
        self.canvas_img1.bind('<Right>', self.img1_next)
        self.canvas_img1.bind('<Left>', self.img1_prev)

        self.canvas_img2.focus_set()
        self.canvas_img2.bind('<Right>', self.img2_next)
        self.canvas_img2.bind('<Left>', self.img2_prev)

        # Key bindings for adding a corresponding point
        self.canvas_img1.bind('<Button-1>', self.corresponding_point_img1)
        self.canvas_img2.bind('<Button-1>', self.corresponding_point_img2)
        self.canvas_minimap.bind('<Button-1>', self.corresponding_point_minimap)

        # Key bindings for adding a new point
        self.canvas_img1.bind('<Button-2>', self.add_point_img1)
        self.canvas_img2.bind('<Button-2>', self.add_point_img2)
        self.canvas_minimap.bind('<Button-2>', self.add_point_minimap)

        # Key bindings for save and restore
        master.bind_all('s', self.save_state)
        master.bind_all('l', self.load_state)
        master.bind_all('d', self.delete_selected_point)
        master.bind_all('f', self.delete_selected_point_in_image)

        # Press f to set current point as a floor point
        self.point_list.bind('f', self.select_floor_point)

        # TODO: Initialization?

        self.update_windows()
        self.get_listbox_selection()
        self.get_scale_selection()
        self.update_point_list()

    def select_img_files(self,):
        img_filepaths = fd.askopenfilenames(
            filetypes=img_filetypes, parent=self.master, title='Choose a file')
        self.img_filepaths = [f for f in img_filepaths]
        # TODO: Might break on Windows?
        self.img_files = [f.split('/')[-1] for f in self.img_filepaths]

        self.idx_minimap = len(self.img_filepaths)

        # For calib functions
        self.n_cam = len(self.img_filepaths)
        self.cam_shape = {}
        for cam_index in range(self.n_cam):
            # self.cam[cam_index] = cv2.imread(self.img_filepaths[cam_index])
            w, h = Image.open(self.img_filepaths[0]).size
            self.cam_shape[cam_index] = (h, w, 3)

        self.file_list_img1.delete(0, tk.END)
        self.file_list_img2.delete(0, tk.END)
        # Write into file lists
        for idx, f_name in enumerate(self.img_files):
            self.file_list_img1.insert(
                    tk.END, '{} : {}\n'.format(idx, f_name))
            self.file_list_img2.insert(
                    tk.END, '{} : {}\n'.format(idx, f_name))

        self.file_list_img1.update()
        self.file_list_img2.update()
        
        self.update_text_dialogs()
        self.redraw_images()
        self.redraw_points()

    def select_minimap(self,):
        self.minimap_path = fd.askopenfilename(filetypes=img_filetypes)
        # TODO: Might break on Windows?
        self.minimap_filename = self.minimap_path.split('/')[-1]
        # self.minimap_cv2 = cv2.imread(self.minimap_path)
        w, h = Image.open(self.minimap_path).size
        self.minimap_shape = (h, w, 3)
        
        self.idx_minimap = len(self.img_filepaths)

        self.redraw_minimap()
        self.redraw_points()

    def add_point_img1(self, event):  # on click
        # Get current point labels
        point_labels = []
        for img_lbl in self.point_data.keys():
            point_labels += [int(i) for i in [*self.point_data[img_lbl]]]
        point_labels = sorted(set(point_labels))


        if len(point_labels) == 0:
            new_point_label = 0
        else:
            new_point_label = max(point_labels) + 1

        for idx, p_lbl in enumerate(point_labels):
            if idx != p_lbl:
                new_point_label = idx
                break

        scale = float(self.img_scale.get())
        x = int(event.x * scale)
        y = int(event.y * scale)

        if not self.point_data.get(self.idx_img1):
            self.point_data[self.idx_img1] = {}
        self.point_data[self.idx_img1][new_point_label] = (x, y)
        self.update_point_list()
        self.redraw_points()

    def add_point_img2(self, event):  # on click
        # Get current point labels
        point_labels = []
        for img_lbl in self.point_data.keys():
            point_labels += [int(i) for i in [*self.point_data[img_lbl]]]
        point_labels = sorted(set(point_labels))

        if len(point_labels) == 0:
            new_point_label = 0
        else:
            new_point_label = max(point_labels) + 1

        for idx, p_lbl in enumerate(point_labels):
            if idx != p_lbl:
                new_point_label = idx
                break

        scale = float(self.img_scale.get())
        x = int(event.x * scale)
        y = int(event.y * scale)

        if not self.point_data.get(self.idx_img2):
            self.point_data[self.idx_img2] = {}
        self.point_data[self.idx_img2][new_point_label] = (x, y)
        self.update_point_list()
        self.redraw_points()
    
    def add_point_minimap(self, event):  # on click
        # Get current point labels
        point_labels = []
        for img_lbl in self.point_data.keys():
            point_labels += [int(i) for i in [*self.point_data[img_lbl]]]
        point_labels = sorted(set(point_labels))

        if len(point_labels) == 0:
            new_point_label = 0
        else:
            new_point_label = max(point_labels) + 1

        for idx, p_lbl in enumerate(point_labels):
            if idx != p_lbl:
                new_point_label = idx
                break

        scale = float(self.img_scale.get())
        x = int(event.x * scale)
        y = int(event.y * scale)

        if not self.point_data.get(self.idx_minimap):
            self.point_data[self.idx_minimap] = {}
        self.point_data[self.idx_minimap][new_point_label] = (x, y)
        self.update_point_list()
        self.redraw_points()

    def delete_selected_point(self, event):
        # Delete point data from self.point_data
        for idx in self.point_data.keys():
            if self.point_lbl in self.point_data[idx].keys():
                del self.point_data[idx][self.point_lbl]
        
        # Delete from point list
        self.point_list.delete(self.point_idx)

        # Delete from floor_points
        if self.point_lbl in self.floor_points:
            self.floor_points.remove(self.point_lbl)


        # Update the point label
        point_lbls = [int(i.strip()) for i in self.point_list.get(0, tk.END)]
        self.point_idx = min(self.point_idx, len(point_lbls) - 1)
        self.point_lbl = point_lbls[self.point_idx]

        self.update_point_list()
        self.redraw_points()
    
    def delete_selected_point_in_image(self, event):
        # Get the active window
        focus = str(self.master.focus_get())
        if focus == '.!toplevel.!canvas':
            # Img 1
            img_idx = self.idx_img1
        elif focus == '.!toplevel2.!canvas':
            # Img 1
            img_idx = self.idx_img2
        elif focus == '.!toplevel3.!canvas':
            img_idx = self.idx_minimap
        else:
            return
        
        # Delete the point from the view
        del self.point_data[img_idx][self.point_lbl]
        
        # Check if self.point_lbl in any point data
        point_labels = []
        for idx in self.point_data.keys():
            point_labels += self.point_data[idx].keys()
        point_labels = sorted(set(point_labels))
        
        # If this point is not existent, then clean up
        if not self.point_lbl in point_labels:
            # Delete from point list
            self.point_list.delete(self.point_idx)

            # Delete from floor_points
            if self.point_lbl in self.floor_points:
                self.floor_points.remove(self.point_lbl)

            # Update the point label
            point_lbls = [int(i.strip()) for i in self.point_list.get(0, tk.END)]
            self.point_idx = min(self.point_idx, len(point_lbls) - 1)
            self.point_lbl = point_lbls[self.point_idx]

        self.update_point_list()
        self.redraw_points()

    def corresponding_point_img1(self, event):  # on click
        scale = float(self.img_scale.get())
        x = int(event.x * scale)
        y = int(event.y * scale)
        if not self.point_data.get(self.idx_img1):
            self.point_data[self.idx_img1] = {}
        self.point_data[self.idx_img1][self.point_lbl] = (x, y)
        self.update_point_list()
        self.redraw_points()

    def corresponding_point_img2(self, event):  # on click
        scale = float(self.img_scale.get())
        x = int(event.x * scale)
        y = int(event.y * scale)
        if not self.point_data.get(self.idx_img2):
            self.point_data[self.idx_img2] = {}
        self.point_data[self.idx_img2][self.point_lbl] = (x, y)
        self.update_point_list()
        self.redraw_points()
    
    def corresponding_point_minimap(self, event):  # on click
        scale = float(self.img_scale.get())
        x = int(event.x * scale)
        y = int(event.y * scale)
        if not self.point_data.get(self.idx_minimap):
            self.point_data[self.idx_minimap] = {}
        self.point_data[self.idx_minimap][self.point_lbl] = (x, y)
        self.update_point_list()
        self.redraw_points()

    def img1_next(self, *args):
        self.idx_img1 += 1
        self.idx_img1 = self.idx_img1 % len(self.img_filepaths)
        self.update_text_dialogs()
        self.redraw_images()
        self.redraw_points()

    def img1_prev(self, *args):
        self.idx_img1 -= 1
        self.idx_img1 = self.idx_img1 % len(self.img_filepaths)
        self.update_text_dialogs()
        self.redraw_images()
        self.redraw_points()

    def img2_next(self, *args):
        self.idx_img2 += 1
        self.idx_img2 = self.idx_img2 % len(self.img_filepaths)
        self.update_text_dialogs()
        self.redraw_images()
        self.redraw_points()

    def img2_prev(self, *args):
        self.idx_img2 -= 1
        self.idx_img2 = self.idx_img2 % len(self.img_filepaths)
        self.update_text_dialogs()
        self.redraw_images()
        self.redraw_points()

    def update_text_dialogs(self):
        for idx, f_name in enumerate(self.img_files):
            if idx == self.idx_img1:
                self.file_list_img1.itemconfig(idx, {'fg': 'red'})
            else:
                self.file_list_img1.itemconfig(idx, {'fg': 'white'})
            if idx == self.idx_img2:
                self.file_list_img2.itemconfig(idx, {'fg': 'red'})
            else:
                self.file_list_img2.itemconfig(idx, {'fg': 'white'})

            if idx in self.calibrated:
                self.file_list_img1.itemconfig(idx, {'bg': 'green'})
                self.file_list_img2.itemconfig(idx, {'bg': 'green'})
            
        self.file_list_img1.update()
        self.file_list_img2.update()

    def update_point_list(self):
        # Get point list entries
        point_list_entries = [int(i.strip()) for i in self.point_list.get(0, tk.END)]
        # Get actual added points
        point_labels = []
        for img_lbl in self.point_data.keys():
            point_labels += [*self.point_data[img_lbl]]
        point_labels = sorted(set(point_labels))

        # First, check if all point_list_entries are in point_lables
        for idx, p_entry in enumerate(point_list_entries):
            if p_entry not in point_labels:
                self.point_list.delete(idx)
        
        point_list_entries = [int(i.strip()) for i in self.point_list.get(0, tk.END)]

        # Second, check if all points from point_labels are in point_list_entries
        for p_idx, p_lbl in enumerate(point_labels):
            if p_lbl not in point_list_entries:
                self.point_list.insert(p_idx, '{}\n'.format(p_lbl))

        for p_idx, p_lbl in enumerate(point_labels):
            if p_lbl == self.point_lbl:
                self.point_list.itemconfig(p_idx, {'fg': 'red'})
            else:
                self.point_list.itemconfig(p_idx, {'fg': 'white'})
            if p_lbl in self.floor_points:
                self.point_list.itemconfig(p_idx, {'bg': 'green'})
            else:
                self.point_list.itemconfig(p_idx, {'bg': ''})

        self.file_list_img1.update()
        self.file_list_img2.update()

    def redraw_images(self):
        # Canvas 1
        self.canvas_img1.delete('img1')
        file_path_img1 = self.img_filepaths[self.idx_img1]
        self.img1 = load_image(
            file_path_img1, scale=float(self.img_scale.get()))
        self.canvas_img1.create_image(
            0, 0, anchor=tk.NW, image=self.img1, tags='img1')
        self.canvas_img1.update()

        # Canvas 2
        self.canvas_img1.delete('img2')
        file_path_img2 = self.img_filepaths[self.idx_img2]
        self.img2 = load_image(
            file_path_img2, scale=float(self.img_scale.get()))
        self.canvas_img2.create_image(
            0, 0, anchor=tk.NW, image=self.img2, tags='img2')
        self.canvas_img2.update()

    def redraw_minimap(self):
        # Canvas 3
        self.canvas_minimap.delete('minimap')
        self.minimap = load_image(
             self.minimap_path, scale=float(self.img_scale.get()))
        self.canvas_minimap.create_image(
            0, 0, anchor=tk.NW, image=self.minimap, tags='minimap')
        self.canvas_minimap.update()

    def redraw_points(self):
        self.canvas_img1.delete('point')
        self.canvas_img2.delete('point')
        self.canvas_minimap.delete('point')

        scale = float(self.img_scale.get())

        for img_lbl in self.point_data.keys():
            pt_labels = [i for i in [*self.point_data[img_lbl]]]
            for p_lbl in pt_labels:
                if img_lbl == self.idx_img1:
                    x = int(self.point_data[img_lbl][p_lbl][0] / scale)
                    y = int(self.point_data[img_lbl][p_lbl][1] / scale)
                    if p_lbl == self.point_lbl:
                        color = 'red'
                    else:
                        color = 'blue'
                    self.canvas_img1.create_oval(
                        x-2, y-2, x+2, y+2, fill=color, width=0, tags='point')
                    self.canvas_img1.create_text(
                        x+5, y-5, text=p_lbl, fill=color, font=('Helvetica 12 bold'), tags='point')

                if img_lbl == self.idx_img2:
                    x = int(self.point_data[img_lbl][p_lbl][0] / scale)
                    y = int(self.point_data[img_lbl][p_lbl][1] / scale)
                    if p_lbl == self.point_lbl:
                        color = 'red'
                    else:
                        color = 'blue'
                    self.canvas_img2.create_oval(
                        x-2, y-2, x+2, y+2, fill=color, width=0, tags='point')
                    self.canvas_img2.create_text(
                        x+5, y-5, text=p_lbl, fill=color, font=('Helvetica 12 bold'), tags='point')
                
                if img_lbl == self.idx_minimap:
                    x = int(self.point_data[img_lbl][p_lbl][0] / scale)
                    y = int(self.point_data[img_lbl][p_lbl][1] / scale)
                    if p_lbl == self.point_lbl:
                        color = 'red'
                    else:
                        color = 'blue'
                    self.canvas_minimap.create_oval(
                        x-2, y-2, x+2, y+2, fill=color, width=0, tags='point')
                    self.canvas_minimap.create_text(
                        x+5, y-5, text=p_lbl, fill=color, font=('Helvetica 12 bold'), tags='point')

        self.canvas_img1.update()
        self.canvas_img2.update()
        self.canvas_minimap.update()

    def save_state(self, *args):
        if self.save_dir == None:
            self.save_dir = fd.askdirectory()
        
        state = {}
        state['img_filepaths'] = self.img_filepaths
        state['img_files'] = self.img_files
        state['minimap_path'] = self.minimap_path
        state['minimap_filename'] = self.minimap_filename
        state['idx_img1'] = self.idx_img1
        state['idx_img2'] = self.idx_img2
        state['point_idx'] = self.point_idx
        state['point_lbl'] = self.point_lbl
        state['floor_points'] = self.floor_points
        state['point_data'] = self.point_data
        state['calibrated'] = self.calibrated
        state['calib_param'] = self.calib_param
        
        filenames = [os.path.splitext(f)[0] for f in self.img_files + [self.minimap_filename]]
        folder_name = "_".join(filenames)

        state_folder_path = os.path.join(self.save_dir, folder_name)
        if not os.path.exists(state_folder_path):
            os.makedirs(state_folder_path)
        
        # Store the state
        state_path = os.path.join(state_folder_path, 'state.npy')
        np.save(state_path, state)

        # Store the point data
        point_data_path = os.path.join(state_folder_path, 'cam.npy')
        np.save(point_data_path, self.point_data)

        # Store the calibration matrices
        for calib_idx in self.calibrated:
            calib_mat_filename = os.path.splitext(self.img_files[calib_idx])[0] + '.npy'
            calib_mat_path = os.path.join(state_folder_path, calib_mat_filename)
            np.save(calib_mat_path, self.calib_param[calib_idx])
    
    def load_state(self, *args):
        # Select folder if not selected
        load_path = fd.askopenfilename(filetypes=npy_filetypes)
        state = np.load(load_path, allow_pickle=True).item()

        # State Vars
        self.img_filepaths = state['img_filepaths']
        self.img_files = state['img_files']
        self.minimap_path = state['minimap_path']
        self.minimap_filename = state['minimap_filename']
        self.idx_img1 = state['idx_img1']
        self.idx_img2 = state['idx_img2']
        self.point_idx = state['point_idx']
        self.point_lbl = state['point_lbl']
        self.floor_points = state['floor_points']
        self.point_data = state['point_data']
        self.calibrated = state['calibrated']
        self.calib_param = state['calib_param']

        self.update_text_dialogs()
        self.redraw_images()
        self.redraw_points()
        self.update_point_list()
    
    def load_point_data(self,):
        # Select folder if not selected
        load_path = fd.askopenfilename(filetypes=npy_filetypes)
        loaded_point_data = np.load(load_path, allow_pickle=True).item()

        # Make sure all keys are represented as int
        point_data = {}
        for cam_key in loaded_point_data.keys():
            point_data[int(cam_key)] = {}
            for point_key in loaded_point_data[cam_key].keys():
                point_data[int(cam_key)][int(point_key)] = loaded_point_data[cam_key][point_key]
        self.point_data = point_data

        self.update_text_dialogs()
        self.redraw_points()
        self.update_point_list()
    
    def load_parameters(self,):
        parameter_filepaths = fd.askopenfilenames(
            filetypes=npy_filetypes, parent=self.master, title='Choose file')
        # TODO: Might break on Windows?
        param_file_names = [f.split('/')[-1] for f in parameter_filepaths]
        file_base = [os.path.splitext(f)[0] for f in self.img_files]

        for param_filepath, param_filename in zip(parameter_filepaths, param_file_names):
            param_file_base = os.path.splitext(param_filename)[0]
            if param_file_base in file_base:
                # Load the parameters
                parameters = np.load(param_filepath)
                # Get the index
                idx = file_base.index(param_file_base)
                # Put them into the self.calib
                self.calib_param[idx] = parameters
                # Add to self.calibrated
                if not idx in self.calibrated:
                    self.calibrated.append(idx)
            else:
                print('Ignoring paramter: {}'.format(param_filename))

        self.update_text_dialogs()
        self.redraw_points()
        self.update_point_list()

    def calibrate_2cams(self,):
        normalized_saved = saved_normalize(self.point_data, self.n_cam, self.cam_shape)
        normalized_point_info, normalized_map_point_info = saved_to_info(normalized_saved, self.n_cam)

        parameters_dict = calib_2cam(normalized_point_info, normalized_map_point_info,
                                        self.idx_img1, self.idx_img2,
                                        self.floor_points, 
                                        self.n_cam)
        # Update the
        for calib_idx in parameters_dict.keys():
            if not calib_idx in self.calibrated:
                self.calibrated.append(calib_idx)
            self.calib_param[calib_idx] = parameters_dict[calib_idx]
    
    def calibrate_ncams(self,):
        normalized_saved = saved_normalize(self.point_data, self.n_cam, self.cam_shape)
        normalized_point_info, normalized_map_point_info = saved_to_info(normalized_saved, self.n_cam)


        parameters_dict = calib_ncam(self.calib_param, 
                                    normalized_saved, 
                                    self.n_cam, 
                                    normalized_map_point_info, 
                                    normalized_point_info, 
                                    self.floor_points)
        
        # Update the
        for calib_idx in parameters_dict.keys():
            if not calib_idx in self.calibrated:
                self.calibrated.append(calib_idx)
            self.calib_param[calib_idx] = parameters_dict[calib_idx]

    def calibrate_1cam(self,):
        normalized_saved = saved_normalize(self.point_data, self.n_cam, self.cam_shape)
        normalized_point_info, normalized_map_point_info = saved_to_info(normalized_saved, self.n_cam)
    
        calibrated_points = get_precal_coords(self.idx_img1,
                                            self.calibrated, 
                                            self.calib_param,
                                            normalized_point_info,
                                            self.floor_points)
        
        parameters = calib_1cam(self.idx_img1, calibrated_points, normalized_map_point_info, self.n_cam, normalized_saved)

        if not self.idx_img1 in self.calibrated:
            self.calibrated.append(self.idx_img1)
        self.calib_param[self.idx_img1] = parameters

    def display_image_points_to_minimap_points(self):
        self.canvas_img1.delete('calib_points')
        self.canvas_img2.delete('calib_points')
        self.canvas_minimap.delete('calib_points')
        
        scale = float(self.img_scale.get())

        if self.idx_img1 in self.calib_param.keys():
            cam_point, ground_point = get_calibrated_points(
                                            parameters=self.calib_param[self.idx_img1],
                                            cam_shape=self.cam_shape[self.idx_img1], 
                                            plane_figure_shape=self.minimap_shape)
        
            # Calibration for camera 1
            color = 'magenta'
            for i in range(cam_point.shape[0]):
                x = int(cam_point[i][0] / scale)
                y = int(cam_point[i][1] / scale)
                self.canvas_img1.create_oval(x-2, y-2, x+2, y+2, fill=color, width=0, tags='calib_points')

                x = int(ground_point[i][0] / scale)
                y = int(ground_point[i][1] / scale)
                self.canvas_minimap.create_oval(x-2, y-2, x+2, y+2, fill=color, width=0, tags='calib_points')
        else:
            print('No Camera calibration available for: {}'.format(self.idx_img1))
        
        if self.idx_img2 in self.calib_param.keys():
            cam_point, ground_point = get_calibrated_points(
                                            parameters=self.calib_param[self.idx_img2],
                                            cam_shape=self.cam_shape[self.idx_img2], 
                                            plane_figure_shape=self.minimap_shape)
            
            # Calibration for camera 2
            color = 'green'
            for i in range(cam_point.shape[0]):
                x = int(cam_point[i][0] / scale)
                y = int(cam_point[i][1] / scale)
                self.canvas_img2.create_oval(x-2, y-2, x+2, y+2, fill=color, width=0, tags='calib_points')

                x = int(ground_point[i][0] / scale)
                y = int(ground_point[i][1] / scale)
                self.canvas_minimap.create_oval(x-2, y-2, x+2, y+2, fill=color, width=0, tags='calib_points')
        else:
            print('No Camera calibration available for: {}'.format(self.idx_img2))
    
    def select_floor_point(self, event):
        if self.point_lbl in self.floor_points:
            self.floor_points.remove(self.point_lbl)
        else:
            self.floor_points.append(self.point_lbl)

    def update_windows(self):
        focus = str(self.master.focus_get())
        if focus == '.':
            self.master.focus_set()
        elif focus == '.!toplevel':
            self.canvas_img1.focus_set()
        elif focus == '.!toplevel2.!canvas':
            self.canvas_img2.focus_set()
        elif focus == '.!toplevel3':
            self.canvas_minimap.focus_set()

        self.after(500, self.update_windows)

    def get_listbox_selection(self):
        l1_selections = self.file_list_img1.curselection()
        if len(l1_selections) > 0:
            if l1_selections != self.previous_l1_selections:
                self.idx_img1 = l1_selections[0]

                self.update_text_dialogs()
                self.redraw_images()
                self.redraw_points()
                self.previous_l1_selections = l1_selections

        l2_selections = self.file_list_img2.curselection()
        if len(l2_selections) > 0:
            if l2_selections != self.previous_l2_selections:
                self.idx_img2 = l2_selections[0]

                self.update_text_dialogs()
                self.redraw_images()
                self.redraw_points()
                self.previous_l2_selections = l2_selections

        point_selections = self.point_list.curselection()
        point_lbls = [int(i.strip()) for i in self.point_list.get(0, tk.END)]
        if len(point_selections) > 0:
            if point_selections != self.previous_point_selections:
                self.point_idx = point_selections[0]
                self.point_lbl = point_lbls[self.point_idx]
                
                self.update_point_list()
                self.redraw_images()
                self.redraw_points()
                self.previous_point_selections = point_selections

        self.after(500, self.get_listbox_selection)

    def get_scale_selection(self):
        scale = float(self.img_scale.get())
        if scale != self.previous_img_scale:
            self.redraw_images()
            self.redraw_minimap()
            self.redraw_points()
            self.previous_img_scale = scale
        
        self.after(500, self.get_scale_selection)
