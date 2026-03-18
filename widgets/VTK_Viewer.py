"""Embeddable VTK 3-D viewer widget.

Wraps a QVTKRenderWindowInteractor inside a QWidget and exposes
high-level methods for displaying polydata meshes, 2-D image slices,
volume renderings, and FreeSurfer surfaces with optional morphometric
colour overlays.
"""

from deps import *
import pyvista as pv
import nibabel as nib

class VTKViewer(QWidget):
    """QWidget hosting a VTK render window for 3-D and slice-based visualisation.

    Supports three display modes:
      - **polydata** -- triangulated surface meshes.
      - **image2d** -- axis-aligned 2-D slices through a vtkImageData volume.
      - **volume** -- GPU-accelerated volume rendering.
    """

    def __init__(self, parent=None):
        """Initialise the VTK viewer and its render window.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self._slice_actor = None
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.vtkWidget)
        self.renderer = vtkRenderer(); self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.07, 0.07, 0.07)
        self._mode = None; self._img = None; self._axis = 2; self._slice = 0; self._slice_min = 0; self._slice_max = 0
        self._slice_mapper = None; self._slice_node = None; self._axes_widget = None   # store widget reference
        self.vtkWidget.Initialize()
        self._init_axes_widget()   # enable orthogonal axes
        
    def show_polydata(self, poly: vtkPolyData):
        """Display a vtkPolyData surface mesh with default lighting.

        Args:
            poly: The polygonal dataset to render.
        """
        self._clear_scene()
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        mapper = vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(0.69, 0.77, 0.87)
        self.renderer.AddActor(actor); self.renderer.ResetCamera()
        self._mode = "polydata"; self.vtkWidget.GetRenderWindow().Render()
        self.show_axes(True)

    def show_polydata_2d(self, poly: vtkPolyData, flat_axis: int = 2):
        """Display a planar vtkPolyData with an orthographic top-down camera.

        Args:
            poly: The polygonal dataset to render.
            flat_axis: The axis perpendicular to the plane (0=X, 1=Y, 2=Z).
        """
        self._clear_scene()
        mapper = vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.69, 0.77, 0.87)
        self.renderer.AddActor(actor)

        # Orthographic camera looking along the flat axis
        cam = self.renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        self.renderer.ResetCamera()

        # Orient camera perpendicular to the flat plane
        pos = list(cam.GetPosition())
        foc = list(cam.GetFocalPoint())
        up = [0, 0, 0]
        if flat_axis == 0:      # flat in X → look along X
            pos[0] = foc[0] + cam.GetDistance()
            up[1] = 1
        elif flat_axis == 1:    # flat in Y → look along Y
            pos[1] = foc[1] + cam.GetDistance()
            up[2] = 1
        else:                   # flat in Z → look along Z
            pos[2] = foc[2] + cam.GetDistance()
            up[1] = 1
        cam.SetPosition(*pos)
        cam.SetViewUp(*up)
        self.renderer.ResetCamera()

        # 2D interaction style (pan + zoom only, no rotation)
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
        self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(vtkInteractorStyleImage())

        self._mode = "polydata2d"
        self.vtkWidget.GetRenderWindow().Render()
        self.show_axes(False)

    def show_image2d(self, img: vtkImageData):
        """Display an axis-aligned 2-D slice through a vtkImageData volume.

        The initial slice is set to the midpoint of the Z-axis extent.

        Args:
            img: The image volume to slice.
        """
        self._clear_scene(); self._img = img
        ex = img.GetExtent(); self._axis = 2
        self._slice_min, self._slice_max = self._axis_minmax(ex, self._axis)
        self._slice = (self._slice_min + self._slice_max)//2
        self._slice_mapper = vtkImageSliceMapper(); self._slice_mapper.SetInputData(img)
        self._apply_orientation_to_mapper(self._axis); self._slice_mapper.SetSliceNumber(self._slice)
        self._slice_node = vtkImageSlice(); self._slice_node.SetMapper(self._slice_mapper)
        mn, mx = img.GetScalarRange();
        if mx <= mn: mx, mn = 1.0, 0.0
        self._slice_node.GetProperty().SetColorWindow(mx-mn)
        self._slice_node.GetProperty().SetColorLevel((mx+mn)*0.5)
        self.renderer.AddViewProp(self._slice_node); self.renderer.ResetCamera()
        self._mode = "image2d"; self.vtkWidget.GetRenderWindow().Render()

    def show_volume(self, img: vtkImageData):
        """Render a vtkImageData volume using GPU smart volume mapping.

        Args:
            img: The image volume to render.
        """
        self._clear_scene()
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        mapper = vtkSmartVolumeMapper(); mapper.SetInputData(img)
        prop = vtkVolumeProperty(); prop.ShadeOn(); prop.SetInterpolationTypeToLinear()
        volume = vtkVolume(); volume.SetMapper(mapper); volume.SetProperty(prop)
        self.renderer.AddVolume(volume); self.renderer.ResetCamera()
        self._mode = "volume"; self.vtkWidget.GetRenderWindow().Render()
        self.show_axes(True)

    def has_slice(self) -> bool:
        """Return True if the viewer is in image-slice mode with valid data."""
        return self._mode == "image2d" and self._img is not None

    def slice_range(self):
        """Return the (min, max) slice index tuple for the current axis."""
        return (self._slice_min, self._slice_max) if self.has_slice() else (0,0)

    def set_slice(self, s: int):
        """Move to slice index *s*, clamped to the valid range."""
        if not self.has_slice(): return
        s = max(self._slice_min, min(self._slice_max, s))
        if s == self._slice: return
        self._slice = s
        if self._slice_mapper: self._slice_mapper.SetSliceNumber(self._slice)
        self.vtkWidget.GetRenderWindow().Render()
    def set_orientation(self, key: str):
        """Change the slicing axis from a human-readable key.

        Args:
            key: One of "Axial (Z)", "Coronal (Y)", or "Sagittal (X)".
        """
        if not self.has_slice(): return
        self._axis = 2 if key=="Axial (Z)" else 1 if key=="Coronal (Y)" else 0
        ex = self._img.GetExtent(); self._slice_min, self._slice_max = self._axis_minmax(ex, self._axis)
        self._slice = max(self._slice_min, min(self._slice_max, self._slice))
        if self._slice_mapper:
            self._apply_orientation_to_mapper(self._axis); self._slice_mapper.SetSliceNumber(self._slice)
        self.vtkWidget.GetRenderWindow().Render()
    def slice_index_to_mm(self, index: int | None = None) -> float:
        """Convert a slice index to a physical position in millimetres.

        Args:
            index: Slice index; defaults to the current slice.

        Returns:
            Physical coordinate along the active axis.
        """
        if not self.has_slice(): return 0.0
        if index is None: index = self._slice
        origin = self._img.GetOrigin(); spacing = self._img.GetSpacing()
        return origin[self._axis] + index * spacing[self._axis]
    def _clear_scene(self):
        """Remove all actors and props from the renderer, keeping the render window alive."""
        rw = self.vtkWidget.GetRenderWindow()

        if not hasattr(self, "renderer") or self.renderer is None:
            from vtkmodules.vtkRenderingCore import vtkRenderer
            self.renderer = vtkRenderer()
            rw.AddRenderer(self.renderer)
            self.renderer.SetBackground(0.07, 0.07, 0.07)
        else:
            # just clear existing actors, keep renderer and axes widget
            self.renderer.RemoveAllViewProps()

        self._slice_mapper = None
        self._slice_node = None
        self._img = None
    @staticmethod
    def _axis_minmax(extent, axis):
        """Return the (min, max) extent pair for the given axis index."""
        return (extent[0],extent[1]) if axis==0 else (extent[2],extent[3]) if axis==1 else (extent[4],extent[5])

    def _apply_orientation_to_mapper(self, axis):
        """Set the slice mapper orientation to match the given axis index."""
        if axis==2: self._slice_mapper.SetOrientationToZ()
        elif axis==1: self._slice_mapper.SetOrientationToY()
        else: self._slice_mapper.SetOrientationToX()


    def _init_axes_widget(self):
        """Create the orientation-marker widget (XYZ triad) in the bottom-left corner."""
        axes = vtkAxesActor()

        w = vtkOrientationMarkerWidget()
        w.SetOrientationMarker(axes)

        # use the interactor of the render window
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        w.SetInteractor(interactor)

        w.SetViewport(0.0, 0.0, 0.2, 0.2)  # bottom-left
        w.SetEnabled(1)
        w.InteractiveOff()

        self._axes_widget = w
        
    def show_axes(self, visible: bool):
        """Toggle visibility of the orientation-marker axes widget."""
        if self._axes_widget is not None:
            self._axes_widget.SetEnabled(1 if visible else 0)
            self.vtkWidget.GetRenderWindow().Render()
        
    def show_slice_with_mesh(self, mesh_file: str, slice_file: str, slice_value: int):
        """Merge original mesh and the selected slice, then show using show_polydata()."""
        # Load data
        mesh = pv.read(mesh_file)
        slices = pv.read(slice_file)

        if "slice_idx" not in slices.point_data:
            print("slice_idx array missing.")
            return

        mask = slices["slice_idx"] == slice_value
        if not np.any(mask):
            print(f"slice_idx {slice_value} not found.")
            return

        # Extract chosen slice
        highlight = slices.extract_points(mask, adjacent_cells=True)

        # Combine both into one PolyData
        combined = pv.merge([mesh, highlight])

        # --- brain mesh actor (very transparent, neutral color) ---
        mesh_poly = mesh.extract_surface()
        mesh_mapper = vtkPolyDataMapper()
        mesh_mapper.SetInputData(mesh_poly)
        mesh_actor = vtkActor()
        mesh_actor.SetMapper(mesh_mapper)
        mesh_actor.GetProperty().SetColor(0.5, 0.5, 0.5)      # gray
        mesh_actor.GetProperty().SetOpacity(0.15)              # more transparent
        mesh_actor.GetProperty().SetAmbient(0.1)
        mesh_actor.GetProperty().SetDiffuse(0.8)
        mesh_actor.GetProperty().SetSpecular(0.2)

        # --- slice actor (bright wireframe, thick lines) ---
        slice_poly = highlight.extract_surface()
        slice_mapper = vtkPolyDataMapper()
        slice_mapper.SetInputData(slice_poly)
        slice_actor = vtkActor()
        slice_actor.SetMapper(slice_mapper)

        prop = slice_actor.GetProperty()
        prop.SetColor(1.0, 0.2, 0.2)                           # strong red
        prop.SetOpacity(1.0)
        prop.SetRepresentationToWireframe()                    # only edges
        prop.SetLineWidth(4)
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(1.0, 1.0, 0.0)                       # yellow edges

        # add both
        self._slice_actor = slice_actor
        self.renderer.AddActor(mesh_actor)
        self.renderer.AddActor(slice_actor)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def delete_slice_section(self):
        """Remove the highlighted slice actor from the scene, if present."""
        if self._slice_actor:
            self.renderer.RemoveActor(self._slice_actor)
            self._slice_actor = None
            self.vtkWidget.GetRenderWindow().Render()
            
    def show_pial_surface(self, surf_path: str):
        """Load and display a single FreeSurfer pial surface (e.g. lh.pial or rh.pial)."""
        # 1) load FreeSurfer surface
        verts, faces = nib.freesurfer.read_geometry(surf_path)  # verts: (N,3), faces: (M,3)

        # 2) build vtkPolyData
        points = vtkPoints()
        points.SetNumberOfPoints(verts.shape[0])
        for i, (x, y, z) in enumerate(verts.astype(float)):
            points.SetPoint(i, float(x), float(y), float(z))

        polys = vtkCellArray()
        for tri in faces.astype(np.int64):
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(tri[0]))
            polys.InsertCellPoint(int(tri[1]))
            polys.InsertCellPoint(int(tri[2]))

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)

        # 3) show using your existing pipeline
        self.show_polydata(poly)

    def _fs_surface_to_poly(self, surf_path: str) -> vtkPolyData:
        """Helper: FreeSurfer surface → vtkPolyData."""
        verts, faces = nib.freesurfer.read_geometry(surf_path)  # verts: (N,3), faces: (M,3)

        points = vtkPoints()
        points.SetNumberOfPoints(verts.shape[0])
        for i, (x, y, z) in enumerate(verts.astype(float)):
            points.SetPoint(i, float(x), float(y), float(z))

        polys = vtkCellArray()
        for tri in faces.astype(np.int64):
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(tri[0]))
            polys.InsertCellPoint(int(tri[1]))
            polys.InsertCellPoint(int(tri[2]))

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)
        return poly

    def show_pial_both(self, lh_surf_path: str, rh_surf_path: str):
        """Load and display both pial surfaces (lh + rh) in one view."""
        lh_poly = self._fs_surface_to_poly(lh_surf_path)
        rh_poly = self._fs_surface_to_poly(rh_surf_path)

        app = vtkAppendPolyData()
        app.AddInputData(lh_poly)
        app.AddInputData(rh_poly)
        app.Update()
        both_poly = app.GetOutput()

        self.show_polydata(both_poly)



    def show_freesurfer_morph(self, surf_path: str, morph_path: str):
        """Display a FreeSurfer surface coloured by a per-vertex morphometric overlay.

        Automatically detects the overlay type (thickness, sulcal depth,
        curvature, etc.) from the file extension and adds a scalar bar.

        Args:
            surf_path: Path to a FreeSurfer surface file (e.g. lh.pial).
            morph_path: Path to a FreeSurfer morph-data file (e.g. lh.thickness).
        """
        # --- 1. load surface + morph ---
        verts, faces = nib.freesurfer.read_geometry(surf_path)
        morph = nib.freesurfer.read_morph_data(morph_path)

        if verts.shape[0] != morph.shape[0]:
            print("vertex count mismatch surface vs morph data")
            return

        # --- 2. decide label from extension ---
        ext = os.path.splitext(os.path.basename(morph_path))[1].lstrip(".").lower()

        if ext in ("thickness", "thick"):
            array_name = "thickness"
            title = "Cortical thickness"
        elif ext in ("sulc", "sulcus"):
            array_name = "sulc"
            title = "Sulcal depth"
        elif ext in ("curv", "curve"):
            array_name = "curv"
            title = "Curvature"
        else:
            array_name = "morph"
            title = ext if ext else "Morph"


        points = vtkPoints()
        points.SetNumberOfPoints(verts.shape[0])
        for i, (x, y, z) in enumerate(verts.astype(float)):
            points.SetPoint(i, float(x), float(y), float(z))

        polys = vtkCellArray()
        for tri in faces.astype(np.int64):
            polys.InsertNextCell(3)
            polys.InsertCellPoint(int(tri[0]))
            polys.InsertCellPoint(int(tri[1]))
            polys.InsertCellPoint(int(tri[2]))

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)

        # --- 4. attach morph data with dynamic name ---
        arr = vtkFloatArray()
        arr.SetName(array_name)
        arr.SetNumberOfValues(morph.shape[0])
        for i, v in enumerate(morph.astype(float)):
            arr.SetValue(i, float(v))

        poly.GetPointData().AddArray(arr)
        poly.GetPointData().SetActiveScalars(array_name)

        # --- 5. map + scalar bar ---
        self._clear_scene()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(float(morph.min()), float(morph.max()))
        mapper.ScalarVisibilityOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)

        scalar_bar = vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetTitle(title)
        scalar_bar.GetTitleTextProperty().SetFontSize(12)
        scalar_bar.GetLabelTextProperty().SetFontSize(8)
        self.renderer.AddActor2D(scalar_bar)

        self.renderer.ResetCamera()
        self._mode = "polydata"
        self.vtkWidget.GetRenderWindow().Render()


    def show_freesurfer_morph_both(
        self,
        lh_surf_path: str,
        lh_morph_path: str,
        rh_surf_path: str,
        rh_morph_path: str,
    ):
        """Display both-hemisphere FreeSurfer surfaces with morphometric colour overlay.

        Args:
            lh_surf_path: Path to left-hemisphere surface file.
            lh_morph_path: Path to left-hemisphere morph-data file.
            rh_surf_path: Path to right-hemisphere surface file.
            rh_morph_path: Path to right-hemisphere morph-data file.
        """
        # --- load surfaces + morphs ---
        lh_verts, lh_faces = nib.freesurfer.read_geometry(lh_surf_path)
        rh_verts, rh_faces = nib.freesurfer.read_geometry(rh_surf_path)

        lh_morph = nib.freesurfer.read_morph_data(lh_morph_path)
        rh_morph = nib.freesurfer.read_morph_data(rh_morph_path)

        if lh_verts.shape[0] != lh_morph.shape[0]:
            print("LH: vertex count mismatch surface vs morph data")
            return
        if rh_verts.shape[0] != rh_morph.shape[0]:
            print("RH: vertex count mismatch surface vs morph data")
            return

        # --- decide label / array name from extension (use LH morph) ---
        ext = os.path.splitext(os.path.basename(lh_morph_path))[1].lstrip(".").lower()
        if ext in ("thickness", "thick"):
            array_name = "thickness"
            title = "Cortical thickness"
        elif ext in ("sulc", "sulcus"):
            array_name = "sulc"
            title = "Sulcal depth"
        elif ext in ("curv", "curve"):
            array_name = "curv"
            title = "Curvature"
        else:
            array_name = "morph"
            title = ext if ext else "Morph"

        # helper: build vtkPolyData from verts, faces, morph array
        def build_poly(verts, faces, morph, name: str) -> vtkPolyData:
            points = vtkPoints()
            points.SetNumberOfPoints(verts.shape[0])
            for i, (x, y, z) in enumerate(verts.astype(float)):
                points.SetPoint(i, float(x), float(y), float(z))

            polys = vtkCellArray()
            for tri in faces.astype(np.int64):
                polys.InsertNextCell(3)
                polys.InsertCellPoint(int(tri[0]))
                polys.InsertCellPoint(int(tri[1]))
                polys.InsertCellPoint(int(tri[2]))

            poly = vtkPolyData()
            poly.SetPoints(points)
            poly.SetPolys(polys)

            arr = vtkFloatArray()
            arr.SetName(name)
            arr.SetNumberOfValues(morph.shape[0])
            for i, v in enumerate(morph.astype(float)):
                arr.SetValue(i, float(v))

            poly.GetPointData().AddArray(arr)
            poly.GetPointData().SetActiveScalars(name)
            return poly

        poly_lh = build_poly(lh_verts, lh_faces, lh_morph, array_name)
        poly_rh = build_poly(rh_verts, rh_faces, rh_morph, array_name)

        # --- append both hemis into a single mesh ---
        app = vtkAppendPolyData()
        app.AddInputData(poly_lh)
        app.AddInputData(poly_rh)
        app.Update()
        poly_both = app.GetOutput()

        morph_min = float(min(lh_morph.min(), rh_morph.min()))
        morph_max = float(max(lh_morph.max(), rh_morph.max()))

        # --- clear scene, map, and show ---
        self._clear_scene()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_both)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(array_name)
        mapper.SetScalarRange(morph_min, morph_max)
        mapper.ScalarVisibilityOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)

        # scalar bar
        scalar_bar = vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetTitle(title)
        scalar_bar.GetTitleTextProperty().SetFontSize(12)
        scalar_bar.GetLabelTextProperty().SetFontSize(8)
        self.renderer.AddActor2D(scalar_bar)

        self.renderer.ResetCamera()
        self._mode = "polydata"
        self.vtkWidget.GetRenderWindow().Render()
