"""
MOCO Viewer - A GUI tool for visualizing latent space and generating motion-corrected images.

This module provides a graphical interface for exploring the latent space of a trained
generator and visualizing the resulting motion-corrected images. It allows users
to load HDF5 files containing trained models and interactively generate output images by
navigating through the latent space.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.render import render_volume  # noqa: E402
from evaluation.results import list_runs as list_hdf5_runs  # noqa: E402
from evaluation.results import load_run  # noqa: E402
from model.generator import MotionGenerator  # noqa: E402
from model.spatial_transformer import SpatialTransformer  # noqa: E402,F401


def load_hdf5_data(results_file, run_name='Run0'):
    """Legacy signature: (params, template, training_latent, state_dict)."""
    run = load_run(results_file, run_name)
    return run.params, run.template, run.training_latent, run.state_dict


def generate_image_from_latent(latent_vector, generator, template):
    """Legacy argument order for evaluation.render.render_volume."""
    return render_volume(generator, template, latent_vector)


def _generator_factory(size):
    def build(params):
        return MotionGenerator(size, params['siz_l'], params['gen_base'], params['out_channels'])
    return build


generator3D_64, generator3D_96, generator3D_128, generator3D_256, generator3D_320 = (
    _generator_factory(s) for s in (64, 96, 128, 256, 320))

def normalize_state_dict_for_model(state_dict, model):
    """
    Normalize state_dict keys to match the model's expected prefixing.

    Handles cases where keys are like "conv_blocks.*" vs "module.conv_blocks.*".
    """
    model_keys = list(model.state_dict().keys())
    model_has_module_prefix = any(k.startswith('module.') for k in model_keys)
    state_has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

    # Fast path: prefixes already aligned
    if model_has_module_prefix == state_has_module_prefix:
        return state_dict

    adjusted = {}
    if state_has_module_prefix and not model_has_module_prefix:
        # Strip leading 'module.'
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            adjusted[new_key] = v
    elif not state_has_module_prefix and model_has_module_prefix:
        # Add leading 'module.'
        for k, v in state_dict.items():
            adjusted[f'module.{k}'] = v
    else:
        adjusted = state_dict

    return adjusted

class ImageSliceFrame(ttk.Frame):
    """
    A frame for displaying a single slice of a 3D image.

    This class creates a frame with a plot area for displaying a single slice of a 3D image.
    It also includes a slider for navigating through different slices of the image.
    """
    
    def __init__(self, parent, title):
        """Initialize the image slice frame with a title and parent widget."""
        super().__init__(parent)
        self.title = title
        self.brightness = 0.0
        self.contrast = 1.0
        
        # Create figure and canvas
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        
        # Add slice slider
        self.slice_var = tk.IntVar(value=0)
        self.slice_slider = ttk.Scale(
            self, 
            from_=0, 
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.slice_var,
            command=self.on_slice_change
        )
        
        # Layout
        ttk.Label(self, text=title, font=("TkDefaultFont", 12, "bold")).pack(pady=5)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.slice_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.ax = self.figure.add_subplot(111)
        self.image = None
        
    def update_image(self, image_data, brightness=0.0, contrast=1.0):
        """
        Update the image data displayed in the frame.

        Args:
            image_data: 3D image data to be displayed
            brightness: Brightness adjustment value
            contrast: Contrast adjustment value
        """
        self.image = image_data
        self.brightness = brightness
        self.contrast = contrast
        if self.image is not None:
            max_slice = self.image.shape[0] - 1
            self.slice_slider.configure(to=max_slice)
            self.show_slice(self.slice_var.get())
            
    def show_slice(self, slice_idx):
        """
        Display a single slice of the image data.

        Args:
            slice_idx: Index of the slice to be displayed
        """
        if self.image is not None:
            self.ax.clear()
            
            # Get current slice and ensure it's a numpy array
            slice_data = self.image[int(slice_idx)]
            if torch.is_tensor(slice_data):
                slice_data = slice_data.cpu().numpy()
            
            # Normalize the data to [0,1] range
            slice_min = np.min(slice_data)
            slice_max = np.max(slice_data)
            if slice_max > slice_min:
                slice_data = (slice_data - slice_min) / (slice_max - slice_min)
            
            # Apply contrast and brightness adjustments
            adjusted_data = self.contrast * (slice_data - 0.5) + 0.5 + self.brightness
            
            # Clip values to [0,1] range
            adjusted_data = np.clip(adjusted_data, 0, 1)
            
            self.ax.imshow(adjusted_data, cmap='gray', vmin=0, vmax=1)
            self.ax.axis('off')
            self.canvas.draw()
            
    def on_slice_change(self, event):
        """Update the displayed slice when the slider position changes."""
        self.show_slice(self.slice_var.get())

class MOCOViewer(tk.Tk):
    """
    A GUI application for visualizing MOCO-STORM latent space and generated images.
    
    This class creates a window with controls for:
    - Loading HDF5 files containing trained models
    - Visualizing the latent space components
    - Generating and displaying output images
    - Navigating through different slices of the output
    """
    
    def __init__(self):
        """Initialize the MOCO viewer with all necessary GUI components."""
        super().__init__()
        
        self.title("MOCO-STORM Viewer")
        self.geometry("1200x800")
        
        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls
        controls = ttk.Frame(main_container)
        controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls, text="Load HDF5", command=self.load_hdf5).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        self.export_video_button = ttk.Button(controls, text="Export Video", command=self.open_video_export_dialog)
        self.export_video_button.pack(side=tk.LEFT, padx=5)
        
        # Brightness and contrast controls
        display_controls = ttk.LabelFrame(main_container, text="Display Controls")
        display_controls.pack(fill=tk.X, pady=5)

        # Brightness control
        ttk.Label(display_controls, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.brightness_slider = ttk.Scale(
            display_controls,
            from_=-1.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.brightness_var,
            command=self.on_display_change
        )
        self.brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Contrast control
        ttk.Label(display_controls, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_slider = ttk.Scale(
            display_controls,
            from_=0.1,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.contrast_var,
            command=self.on_display_change
        )
        self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Frame slider control (for multi-frame templates)
        self.frame_control = ttk.LabelFrame(main_container, text="Frame Selection")
        self.frame_control.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.frame_control, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.frame_var = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(
            self.frame_control,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self.on_frame_change
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Hide frame control initially (will show if multi-frame template is loaded)
        self.frame_control.pack_forget()
        
        # Latent plot frame
        latent_frame = ttk.Frame(main_container)
        latent_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.latent_figure = Figure(figsize=(12, 3))
        self.latent_canvas = FigureCanvasTkAgg(self.latent_figure, latent_frame)
        self.latent_canvas.draw()
        self.latent_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.latent_ax = self.latent_figure.add_subplot(111)
        
        # Frame for latent sliders
        self.latent_sliders_frame = ttk.Frame(latent_frame)
        self.latent_sliders_frame.pack(fill=tk.X, padx=5)
        
        # Add global latent slider
        global_latent_frame = ttk.Frame(self.latent_sliders_frame)
        global_latent_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(global_latent_frame, text="Global Latent:").pack(side=tk.LEFT, padx=5)
        self.global_latent_var = tk.IntVar(value=0)
        self.global_latent_slider = ttk.Scale(
            global_latent_frame,
            from_=0,
            to=0,  # Will be updated when latent is loaded
            orient=tk.HORIZONTAL,
            variable=self.global_latent_var,
            command=self.on_global_latent_change
        )
        self.global_latent_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Will store slider variables
        self.latent_vars = []
        self.latent_sliders = []
        
        # Image views
        views_frame = ttk.Frame(main_container)
        views_frame.pack(fill=tk.BOTH, expand=True)
        
        self.axial_view = ImageSliceFrame(views_frame, "Axial View")
        self.sagittal_view = ImageSliceFrame(views_frame, "Sagittal View")
        self.coronal_view = ImageSliceFrame(views_frame, "Coronal View")
        
        self.axial_view.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.sagittal_view.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.coronal_view.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        
        views_frame.grid_columnconfigure(0, weight=1)
        views_frame.grid_columnconfigure(1, weight=1)
        views_frame.grid_columnconfigure(2, weight=1)
        
        # Initialize variables
        self.current_file = None
        self.current_run = None
        self.params = None
        self.template = None
        self.latent = None
        self.state_dict = None
        self.generator = None
        self.transfer = None
        self.current_image = None
        self.is_multi_frame = False
        self.num_frames = 1
        
    def create_latent_sliders(self):
        """Create sliders for each dimension of the latent space"""
        # Clear existing sliders
        for slider in self.latent_sliders:
            slider.destroy()
        self.latent_vars.clear()
        self.latent_sliders.clear()
        
        if self.latent is None:
            print("Warning: Latent vector is None")
            return
            
        # Print debug information about latent shape
        print(f"Latent shape in create_latent_sliders: {self.latent.shape}")
        
        # Handle different latent shapes
        if not hasattr(self.latent, 'shape') or len(self.latent.shape) < 2:
            print(f"Warning: Latent has invalid shape: {self.latent.shape if hasattr(self.latent, 'shape') else 'no shape'}")
            n_dims = 1
            max_idx = 0
        else:
            n_dims = self.latent.shape[1]  # First dimension of latent space
            max_idx = len(self.latent) - 1
            
        print(f"Creating {n_dims} latent sliders with max index {max_idx}")
        
        # Update global latent slider range
        self.global_latent_slider.configure(to=max_idx)
        self.global_latent_var.set(0)
        
        for i in range(n_dims):
            frame = ttk.Frame(self.latent_sliders_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"Latent {i+1}:").pack(side=tk.LEFT, padx=5)
            var = tk.IntVar(value=0)
            self.latent_vars.append(var)
            
            slider = ttk.Scale(
                frame,
                from_=0,
                to=max_idx,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda event, idx=i: self.on_latent_change(event, idx)
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.latent_sliders.append(slider)

    def choose_hdf5_run(self, filename):
        """Ask the user which run group to load from a multi-run results file."""
        runs = list_hdf5_runs(filename)
        if not runs:
            raise ValueError("No Run groups found in selected HDF5 file")
        if len(runs) == 1:
            return runs[0]['name']

        dialog = tk.Toplevel(self)
        dialog.title("Select Run")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        selected_run = tk.StringVar(value=runs[-1]['name'])
        result = {'run_name': None}

        ttk.Label(dialog, text="Select run to display:").grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=10, pady=(10, 4))

        run_labels = []
        run_lookup = {}
        for run in runs:
            label_parts = [run['name']]
            if run.get('RunStart'):
                label_parts.append(f"started {run['RunStart']}")
            if run.get('OptimizeLatent'):
                label_parts.append(f"latent {run['OptimizeLatent']}")
            label = " | ".join(str(part) for part in label_parts)
            run_labels.append(label)
            run_lookup[label] = run['name']

        selected_label = tk.StringVar(value=run_labels[-1])
        combo = ttk.Combobox(dialog, textvariable=selected_label, values=run_labels, state="readonly", width=72)
        combo.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=4)

        def confirm():
            selected_run.set(run_lookup[selected_label.get()])
            result['run_name'] = selected_run.get()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        ttk.Button(dialog, text="Load Run", command=confirm).grid(row=2, column=0, sticky=tk.E, padx=5, pady=10)
        ttk.Button(dialog, text="Cancel", command=cancel).grid(row=2, column=1, sticky=tk.W, padx=5, pady=10)
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.columnconfigure(0, weight=1)
        dialog.columnconfigure(1, weight=1)

        self.wait_window(dialog)
        return result['run_name']
            
    def load_hdf5(self):
        """
        Load data from an HDF5 file containing the trained model and latent vectors.
        Opens a file dialog for selection and initializes the viewer with the loaded data.
        """
        # Open file dialog
        filename = filedialog.askopenfilename(
            title="Select HDF5 file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5 *.5h"), ("All files", "*.*")]
        )
        
        if filename:
            run_name = self.choose_hdf5_run(filename)
            if run_name is None:
                return

            # Load data from HDF5
            self.params, self.template, self.latent, self.state_dict = load_hdf5_data(filename, run_name)
            self.current_file = filename
            self.current_run = run_name
            self.title(f"MOCO-STORM Viewer - {os.path.basename(filename)} ({run_name})")
            
            # Debug print
            print(f"Loaded run: {run_name}")
            print("Loaded parameters:", self.params)
            print(f"Template shape after loading: {self.template.shape}")
            print(f"Latent type: {type(self.latent)}, shape: {self.latent.shape if hasattr(self.latent, 'shape') else 'no shape'}")
            
            # Ensure required parameters exist
            # Ensure latent has a second dimension before accessing it
            if hasattr(self.latent, 'shape') and len(self.latent.shape) >= 2:
                self.params['siz_l'] = self.latent.shape[1]
            else:
                # If latent doesn't have proper shape, default to 1
                print(f"Warning: Latent has unexpected shape: {self.latent.shape if hasattr(self.latent, 'shape') else 'no shape'}")
                self.params['siz_l'] = 1
                
                # Try to reshape if possible
                if hasattr(self.latent, 'shape') and len(self.latent.shape) == 1:
                    try:
                        # Reshape 1D latent to 2D with one feature
                        self.latent = self.latent.reshape(-1, 1)
                        print(f"Reshaped latent to: {self.latent.shape}")
                        self.params['siz_l'] = 1
                    except Exception as e:
                        print(f"Could not reshape latent: {e}")
            
            if 'gen_base' not in self.params:
                self.params['gen_base'] = 32  # default value
                
            if 'out_channels' not in self.params:
                self.params['out_channels'] = 3  # default for 3D motion field
            
            # Determine if template is multi-frame and what format it has
            template_shape = self.template.shape
            print(f"Analyzing template shape: {template_shape}")
            
            # Handle different template formats
            if len(template_shape) == 5:
                # Format likely (F, X, Y, Z, 2) - multi-frame with complex components
                self.is_multi_frame = True
                self.num_frames = template_shape[0]
                image_size = template_shape[2]  # X dimension   
                print(f"Detected complex multi-frame template with {self.num_frames} frames")
                
                # Show frame slider
                self.frame_control.pack(fill=tk.X, pady=5, after=self.contrast_slider.master)
                self.frame_slider.configure(to=self.num_frames-1)
                self.frame_var.set(0)
                
            elif len(template_shape) == 4:
                # This is a single-frame complex template (X, Y, Z, 2)
                self.is_multi_frame = False
                self.frame_control.pack_forget()
                image_size = template_shape[0]
                print(f"Detected single-frame complex template")
            
            print(f"Using image size: {image_size}")
            
            # Setup generator
            generator_classes = {
                64: generator3D_64,
                96: generator3D_96,
                128: generator3D_128,
                256: generator3D_256,
                320: generator3D_320
            }
            
            if image_size not in generator_classes:
                print(f"Template shape: {self.template.shape}")
                raise ValueError(f"Unsupported image size: {image_size}. Must be one of {list(generator_classes.keys())}")
            
            self.generator = generator_classes[image_size](self.params)
            # Normalize state dict keys to match model (handle DataParallel 'module.' prefix)
            try:
                adjusted_state = normalize_state_dict_for_model(self.state_dict, self.generator)
                self.generator.load_state_dict(adjusted_state)
            except RuntimeError as e:
                print(f"Strict load failed with: {e}\nRetrying with strict=False")
                self.generator.load_state_dict(adjusted_state, strict=False)
            self.generator.cuda()
            self.generator.eval()
            
            # Convert latent to torch tensor and move to GPU
            if self.latent is not None:
                if isinstance(self.latent, np.ndarray):
                    print("Converting numpy latent to torch tensor")
                    # Make sure it's 2D
                    if len(self.latent.shape) == 1:
                        self.latent = self.latent.reshape(-1, 1)
                        
                    # Reshape latent to (n_states, channels, 1, 1, 1)
                    self.latent = torch.from_numpy(self.latent).float()
                    if len(self.latent.shape) == 2:
                        n_states, channels = self.latent.shape
                        print(f"Reshaping latent from {self.latent.shape} to {(n_states, channels, 1, 1, 1)}")
                        self.latent = self.latent.view(n_states, channels, 1, 1, 1)
                    self.latent = self.latent.cuda()
                elif torch.is_tensor(self.latent):
                    print("Latent is already a torch tensor, ensuring proper shape")
                    # Ensure it's on the right device
                    self.latent = self.latent.to('cuda')
                    
                    # Ensure it has the right shape
                    if len(self.latent.shape) == 1:
                        # Convert 1D to 2D with one feature
                        self.latent = self.latent.view(-1, 1)
                    
                    if len(self.latent.shape) == 2:
                        # Convert 2D to 5D for generator
                        n_states, channels = self.latent.shape
                        print(f"Reshaping tensor latent from {self.latent.shape} to {(n_states, channels, 1, 1, 1)}")
                        self.latent = self.latent.view(n_states, channels, 1, 1, 1)
                else:
                    print(f"WARNING: Latent has unhandled type: {type(self.latent)}")
            
            # Setup transfer
            self.transfer = SpatialTransformer((image_size, image_size, image_size))
            
            # Create sliders for each latent dimension
            self.create_latent_sliders()
            
            # Update latent plot
            self.update_latent_plot()
            
            # Generate initial image
            self.update_output_image()
            
    def update_latent_plot(self):
        """Update the latent plot with current selections"""
        if self.latent is None:
            return
            
        self.latent_ax.clear()
        latent_plot = self.latent.view(len(self.latent), -1).cpu().numpy()
        
        # Plot each latent component with viewer-friendly labels.
        for i in range(self.params['siz_l']):
            self.latent_ax.plot(latent_plot[:, i], label=f'Latent {i+1}')
            
        # Add vertical lines for selected indices without crowding the legend.
        colors = ['r', 'g', 'b', 'c', 'm']  # Different colors for each dimension
        for i, var in enumerate(self.latent_vars):
            idx = var.get()
            self.latent_ax.axvline(x=idx, color=colors[i % len(colors)], linestyle='--')
            
        self.latent_ax.set_xlabel("Frames")
        self.latent_ax.set_ylabel("Amplitude")
        self.latent_ax.legend()
        self.latent_figure.tight_layout()
        self.latent_canvas.draw()

    def get_generation_device(self):
        """Return the device used by the generator."""
        generation_model = self.generator
        if generation_model is not None and hasattr(generation_model, "parameters"):
            try:
                return next(generation_model.parameters()).device
            except StopIteration:
                pass
        return torch.device("cuda")

    def get_latent_value(self, frame_idx, latent_dim):
        """Read one latent value from the stored latent tensor."""
        idx = min(max(int(frame_idx), 0), len(self.latent) - 1)
        if len(self.latent.shape) == 5:
            return self.latent[idx, latent_dim, 0, 0, 0]
        if len(self.latent.shape) == 2:
            return self.latent[idx, latent_dim]
        print(f"WARNING: Unexpected latent shape: {self.latent.shape}")
        return self.latent[idx, latent_dim]

    def build_latent_vector(self, frame_idx=None, variable_dims=None, start_frame_idx=None):
        """
        Build a generator input latent vector.

        If variable_dims is provided, dimensions in that set use frame_idx while
        all other dimensions use start_frame_idx.
        """
        latent_dim = self.params.get('siz_l', 1)
        device = self.get_generation_device()
        z_in = torch.zeros((1, latent_dim, 1, 1, 1), device=device)

        for i in range(latent_dim):
            if i >= self.latent.shape[1]:
                continue

            if variable_dims is None:
                idx = self.latent_vars[i].get() if i < len(self.latent_vars) else 0
            else:
                idx = frame_idx if i in variable_dims else start_frame_idx

            z_in[0, i] = self.get_latent_value(idx, i).to(device)

        return z_in

    def get_template_for_frame(self, frame_idx):
        """Return the template tensor for a requested export/viewer frame."""
        if self.is_multi_frame:
            template_idx = min(max(int(frame_idx), 0), self.num_frames - 1)
            template_frame = self.template[template_idx]
        else:
            template_frame = self.template

        return torch.from_numpy(template_frame).to(self.get_generation_device())

    def normalize_display_slice(self, slice_data, window=None):
        """Apply the same per-slice normalization and display settings as the GUI."""
        if torch.is_tensor(slice_data):
            slice_data = slice_data.cpu().numpy()

        slice_data = np.asarray(slice_data)
        if window is None:
            slice_min = np.min(slice_data)
            slice_max = np.max(slice_data)
        else:
            slice_min, slice_max = window

        if slice_max > slice_min:
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)

        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        adjusted_data = contrast * (slice_data - 0.5) + 0.5 + brightness
        return np.clip(adjusted_data, 0, 1)

    def get_selected_slice_index(self, plane):
        """Return the currently selected slice index for a plane."""
        plane_views = {
            "axial": self.axial_view,
            "sagittal": self.sagittal_view,
            "coronal": self.coronal_view,
        }
        return plane_views[plane].slice_var.get()

    def get_plane_image(self, image_data, plane):
        """Return a 3D image reoriented for display in one viewer plane."""
        if plane == "axial":
            return image_data
        if plane == "sagittal":
            sagittal = np.transpose(image_data, (2, 1, 0))
            sagittal = np.flip(sagittal, axis=1)
            return np.rot90(sagittal, k=-1, axes=(1, 2))
        if plane == "coronal":
            coronal = np.transpose(image_data, (1, 2, 0))
            coronal = np.flip(coronal, axis=1)
            return np.rot90(coronal, k=-1, axes=(1, 2))
        raise ValueError(f"Unknown slice plane: {plane}")

    def extract_display_slice(self, image_data, plane, slice_idx):
        """Extract and normalize the requested plane/slice from a 3D image."""
        return self.normalize_display_slice(self.extract_raw_display_slice(image_data, plane, slice_idx))

    def extract_raw_display_slice(self, image_data, plane, slice_idx):
        """Extract the requested oriented plane/slice without intensity normalization."""
        plane_data = self.get_plane_image(image_data, plane)
        slice_idx = min(max(int(slice_idx), 0), plane_data.shape[0] - 1)
        return np.asarray(plane_data[slice_idx])
    
    def update_output_image(self):
        """
        Generate and display a new output image based on the current latent vector.
        This is called whenever any latent slider position changes.
        """
        if self.generator is None or not self.latent_vars:
            return
        
        # Check if latent has valid shape
        if not hasattr(self.latent, 'shape') or len(self.latent.shape) < 2:
            print("Cannot update image: latent vector has invalid shape")
            return
            
        try:
            z_in = self.build_latent_vector()
        except Exception as e:
            print(f"Error creating latent vector: {e}")
            print(f"Latent shape: {self.latent.shape if hasattr(self.latent, 'shape') else 'no shape'}")
            return
        
        template_display = self.get_template_for_frame(self.frame_var.get())
        
        # Generate output image
        self.current_image = generate_image_from_latent(z_in, self.generator, template_display)
        
        # Update all views
        self.update_all_views()
        
        # Update latent plot
        self.update_latent_plot()
    
    def on_latent_change(self, event, slider_idx):
        """Update the output image when any latent slider position changes."""
        self.update_output_image()
        
    def on_frame_change(self, event):
        """Update the output image when the frame slider position changes."""
        self.update_output_image()
        
    def update_all_views(self):
        """Update all image views with the current output image."""
        if self.current_image is None:
            return
            
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
            
        self.axial_view.update_image(self.get_plane_image(self.current_image, "axial"), brightness, contrast)
        self.sagittal_view.update_image(self.get_plane_image(self.current_image, "sagittal"), brightness, contrast)
        self.coronal_view.update_image(self.get_plane_image(self.current_image, "coronal"), brightness, contrast)
    
    def save_image(self):
        """
        Save the current output image to a file.

        Opens a file dialog for selection of the output file.
        """
        if self.current_image is not None:
            file_name = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".nii",
                filetypes=[("NIFTI files", "*.nii"), ("All files", "*.*")]
            )
            if file_name:
                nifti_img = nib.Nifti1Image(self.current_image, np.eye(4))
                nib.save(nifti_img, file_name)

    def open_video_export_dialog(self):
        """Open the video export options dialog."""
        if self.generator is None or self.latent is None or not self.latent_vars:
            messagebox.showerror("Export Video", "Load an HDF5 file before exporting video.")
            return

        max_frame_idx = len(self.latent) - 1
        if max_frame_idx < 0:
            messagebox.showerror("Export Video", "No latent frames are available to export.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Export Video")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        plane_var = tk.StringVar(value="axial")
        start_var = tk.IntVar(value=0)
        end_var = tk.IntVar(value=max_frame_idx)
        fps_var = tk.DoubleVar(value=12.0)
        output_var = tk.StringVar(value="")
        latent_dim = self.params.get('siz_l', len(self.latent_vars))
        variable_vars = []

        ttk.Label(dialog, text="Slice plane:").grid(row=0, column=0, sticky=tk.W, padx=8, pady=4)
        plane_box = ttk.Combobox(
            dialog,
            textvariable=plane_var,
            values=("axial", "coronal", "sagittal"),
            state="readonly",
            width=12,
        )
        plane_box.grid(row=0, column=1, sticky=tk.EW, padx=8, pady=4)

        ttk.Label(dialog, text=f"Start frame (0-{max_frame_idx}):").grid(row=1, column=0, sticky=tk.W, padx=8, pady=4)
        ttk.Spinbox(dialog, from_=0, to=max_frame_idx, textvariable=start_var, width=10).grid(
            row=1, column=1, sticky=tk.EW, padx=8, pady=4
        )

        ttk.Label(dialog, text=f"End frame (0-{max_frame_idx}):").grid(row=2, column=0, sticky=tk.W, padx=8, pady=4)
        ttk.Spinbox(dialog, from_=0, to=max_frame_idx, textvariable=end_var, width=10).grid(
            row=2, column=1, sticky=tk.EW, padx=8, pady=4
        )

        ttk.Label(dialog, text="Frame rate:").grid(row=3, column=0, sticky=tk.W, padx=8, pady=4)
        ttk.Spinbox(dialog, from_=0.1, to=120.0, increment=1.0, textvariable=fps_var, width=10).grid(
            row=3, column=1, sticky=tk.EW, padx=8, pady=4
        )

        ttk.Label(dialog, text="Output file:").grid(row=4, column=0, sticky=tk.W, padx=8, pady=4)
        output_frame = ttk.Frame(dialog)
        output_frame.grid(row=4, column=1, sticky=tk.EW, padx=8, pady=4)
        ttk.Entry(output_frame, textvariable=output_var, width=36).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def choose_output_file():
            file_name = filedialog.asksaveasfilename(
                parent=dialog,
                title="Export Video",
                defaultextension=".mp4",
                filetypes=[("MPEG-4 video", "*.mp4"), ("All files", "*.*")],
            )
            if file_name:
                output_var.set(file_name)

        ttk.Button(output_frame, text="Browse...", command=choose_output_file).pack(side=tk.LEFT, padx=(5, 0))

        latent_frame = ttk.LabelFrame(dialog, text="Latents to Vary")
        latent_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=8, pady=8)
        for i in range(latent_dim):
            var = tk.BooleanVar(value=True)
            variable_vars.append(var)
            ttk.Checkbutton(latent_frame, text=f"Latent {i + 1}", variable=var).grid(
                row=i // 4,
                column=i % 4,
                sticky=tk.W,
                padx=6,
                pady=2,
            )

        buttons = ttk.Frame(dialog)
        buttons.grid(row=6, column=0, columnspan=2, sticky=tk.E, padx=8, pady=8)

        def submit_export():
            try:
                start_frame = int(start_var.get())
                end_frame = int(end_var.get())
                fps = float(fps_var.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Export Video", "Frame range and frame rate must be numeric.", parent=dialog)
                return

            if start_frame < 0 or end_frame < 0 or start_frame > max_frame_idx or end_frame > max_frame_idx:
                messagebox.showerror("Export Video", f"Frame range must be within 0-{max_frame_idx}.", parent=dialog)
                return
            if end_frame < start_frame:
                messagebox.showerror("Export Video", "End frame must be greater than or equal to start frame.", parent=dialog)
                return
            if fps <= 0:
                messagebox.showerror("Export Video", "Frame rate must be greater than zero.", parent=dialog)
                return

            output_path = output_var.get().strip()
            if not output_path:
                messagebox.showerror("Export Video", "Choose an output file path.", parent=dialog)
                return
            if not output_path.lower().endswith(".mp4"):
                output_path += ".mp4"

            settings = {
                "plane": plane_var.get(),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "output_path": output_path,
                "variable_dims": {i for i, var in enumerate(variable_vars) if var.get()},
            }
            dialog.destroy()
            self.export_video(settings)

        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons, text="Export", command=submit_export).pack(side=tk.RIGHT, padx=5)
        dialog.columnconfigure(1, weight=1)

    def generate_video_slice(self, frame_idx, start_frame_idx, variable_dims, plane, slice_idx, window=None):
        """Generate one normalized 2D video frame."""
        raw_slice = self.generate_video_raw_slice(frame_idx, start_frame_idx, variable_dims, plane, slice_idx)
        return self.normalize_display_slice(raw_slice, window=window)

    def generate_video_raw_slice(self, frame_idx, start_frame_idx, variable_dims, plane, slice_idx):
        """Generate one raw oriented 2D video slice."""
        z_in = self.build_latent_vector(
            frame_idx=frame_idx,
            variable_dims=variable_dims,
            start_frame_idx=start_frame_idx,
        )
        template_display = self.get_template_for_frame(frame_idx)
        image_data = generate_image_from_latent(z_in, self.generator, template_display)
        return self.extract_raw_display_slice(image_data, plane, slice_idx)

    def export_video(self, settings):
        """Generate and write a high-quality MPEG-4 video."""
        if not animation.writers.is_available("ffmpeg"):
            messagebox.showerror(
                "Export Video",
                "Matplotlib cannot find ffmpeg. Install ffmpeg or make it available on PATH to export MPEG-4 video.",
            )
            return

        output_dir = os.path.dirname(settings["output_path"])
        if output_dir and not os.path.isdir(output_dir):
            messagebox.showerror("Export Video", f"Output directory does not exist:\n{output_dir}")
            return

        plane = settings["plane"]
        start_frame = settings["start_frame"]
        end_frame = settings["end_frame"]
        fps = settings["fps"]
        variable_dims = settings["variable_dims"]
        slice_idx = self.get_selected_slice_index(plane)
        frame_count = end_frame - start_frame + 1

        progress = tk.Toplevel(self)
        progress.title("Exporting Video")
        progress.transient(self)
        progress.resizable(False, False)
        progress_label = ttk.Label(progress, text="Preparing video export...")
        progress_label.pack(padx=20, pady=(15, 8))
        progress_bar = ttk.Progressbar(progress, length=320, mode="determinate", maximum=frame_count)
        progress_bar.pack(padx=20, pady=(0, 15))
        self.export_video_button.configure(state=tk.DISABLED)
        self.update_idletasks()

        try:
            raw_frames = []
            window_min = np.inf
            window_max = -np.inf
            for output_idx, frame_idx in enumerate(range(start_frame, end_frame + 1), start=1):
                raw_frame = self.generate_video_raw_slice(frame_idx, start_frame, variable_dims, plane, slice_idx)
                raw_frames.append(raw_frame)
                window_min = min(window_min, np.min(raw_frame))
                window_max = max(window_max, np.max(raw_frame))
                progress_bar["value"] = output_idx
                progress_label.configure(text=f"Computing window from frame {output_idx} of {frame_count}...")
                self.update_idletasks()

            window = (window_min, window_max)
            first_frame = self.normalize_display_slice(raw_frames[0], window=window)
            height, width = first_frame.shape
            dpi = 100
            figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            FigureCanvasAgg(figure)
            axis = figure.add_axes([0, 0, 1, 1])
            image_artist = axis.imshow(first_frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.axis("off")

            writer = animation.FFMpegWriter(
                fps=fps,
                codec="mpeg4",
                bitrate=50000,
                extra_args=["-q:v", "2", "-pix_fmt", "yuv420p"],
            )

            progress_bar["value"] = 0
            progress_label.configure(text="Writing video frames...")
            self.update_idletasks()

            with writer.saving(figure, settings["output_path"], dpi=dpi):
                for output_idx, raw_frame in enumerate(raw_frames, start=1):
                    if output_idx == 1:
                        video_frame = first_frame
                    else:
                        video_frame = self.normalize_display_slice(raw_frame, window=window)

                    image_artist.set_data(video_frame)
                    writer.grab_frame()
                    progress_bar["value"] = output_idx
                    progress_label.configure(text=f"Exporting frame {output_idx} of {frame_count}...")
                    self.update_idletasks()

            messagebox.showinfo("Export Video", f"Video exported to:\n{settings['output_path']}")
        except Exception as e:
            messagebox.showerror("Export Video", f"Could not export video:\n{e}")
        finally:
            self.export_video_button.configure(state=tk.NORMAL)
            progress.destroy()
    
    def on_display_change(self, event):
        """Update all views when brightness or contrast changes."""
        self.update_all_views()

    def on_global_latent_change(self, event):
        """Update all latent sliders when the global latent slider changes."""
        if not self.latent_vars:
            return
            
        # Get the current value of the global slider
        global_idx = self.global_latent_var.get()
        
        # Update all individual latent sliders
        for var in self.latent_vars:
            var.set(global_idx)
            
        # Update the output image
        self.update_output_image()

if __name__ == '__main__':
    viewer = MOCOViewer()
    viewer.mainloop()
