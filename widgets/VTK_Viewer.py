from deps import *

class VTKViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.vtkWidget)
        self.renderer = vtkRenderer(); self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.07, 0.07, 0.07)
        self._mode = None; self._img = None; self._axis = 2; self._slice = 0; self._slice_min = 0; self._slice_max = 0
        self._slice_mapper = None; self._slice_node = None
        self.vtkWidget.Initialize()

    def show_polydata(self, poly: vtkPolyData):
        self._clear_scene()
        mapper = vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(0.69, 0.77, 0.87)
        self.renderer.AddActor(actor); self.renderer.ResetCamera()
        self._mode = "polydata"; self.vtkWidget.GetRenderWindow().Render()

    def show_image2d(self, img: vtkImageData):
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
        self._clear_scene()
        mapper = vtkSmartVolumeMapper(); mapper.SetInputData(img)
        prop = vtkVolumeProperty(); prop.ShadeOn(); prop.SetInterpolationTypeToLinear()
        volume = vtkVolume(); volume.SetMapper(mapper); volume.SetProperty(prop)
        self.renderer.AddVolume(volume); self.renderer.ResetCamera()
        self._mode = "volume"; self.vtkWidget.GetRenderWindow().Render()

    def has_slice(self) -> bool: return self._mode == "image2d" and self._img is not None
    def slice_range(self): return (self._slice_min, self._slice_max) if self.has_slice() else (0,0)
    def set_slice(self, s: int):
        if not self.has_slice(): return
        s = max(self._slice_min, min(self._slice_max, s))
        if s == self._slice: return
        self._slice = s
        if self._slice_mapper: self._slice_mapper.SetSliceNumber(self._slice)
        self.vtkWidget.GetRenderWindow().Render()
    def set_orientation(self, key: str):
        if not self.has_slice(): return
        self._axis = 2 if key=="Axial (Z)" else 1 if key=="Coronal (Y)" else 0
        ex = self._img.GetExtent(); self._slice_min, self._slice_max = self._axis_minmax(ex, self._axis)
        self._slice = max(self._slice_min, min(self._slice_max, self._slice))
        if self._slice_mapper:
            self._apply_orientation_to_mapper(self._axis); self._slice_mapper.SetSliceNumber(self._slice)
        self.vtkWidget.GetRenderWindow().Render()
    def slice_index_to_mm(self, index: int | None = None) -> float:
        if not self.has_slice(): return 0.0
        if index is None: index = self._slice
        origin = self._img.GetOrigin(); spacing = self._img.GetSpacing()
        return origin[self._axis] + index * spacing[self._axis]
    def _clear_scene(self):
        rw = self.vtkWidget.GetRenderWindow()
        for r in list(rw.GetRenderers()): rw.RemoveRenderer(r)
        self.renderer = vtkRenderer(); rw.AddRenderer(self.renderer); self.renderer.SetBackground(0.07, 0.07, 0.07)
        self._slice_mapper = None; self._slice_node = None; self._img = None
    @staticmethod
    def _axis_minmax(extent, axis): return (extent[0],extent[1]) if axis==0 else (extent[2],extent[3]) if axis==1 else (extent[4],extent[5])
    def _apply_orientation_to_mapper(self, axis):
        if axis==2: self._slice_mapper.SetOrientationToZ()
        elif axis==1: self._slice_mapper.SetOrientationToY()
        else: self._slice_mapper.SetOrientationToX()
