import vtk
import numpy as np
from vtk.util import numpy_support
import math
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import fmin
#import matplotlib.pyplot as plt
import xarray as xr
from itertools import product

def main():
    #createDEM_v1()
    #createDEM_v2()
    add_scalar('TS', '2017-07-01 12:00:00')

def createDEM_v1():

    ds = xr.open_dataset('../data/output/Peru_20160601-20180530_comp4.nc')
    
    points = vtk.vtkPoints()
    
    numPoints = ds.south_north.size*ds.west_east.size
   
    print('Write points \n')
    for i,j in product(ds.south_north.values,ds.west_east.values):
            points.InsertNextPoint(ds.lat.isel(south_north=i,west_east=j), ds.lon.isel(south_north=i,west_east=j), ds.HGT.isel(south_north=i,west_east=j).values/6370000.0)
    
    print('Create unstructured grid \n') 
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(polydata)
    delaunay.Update()

#    subdivision = vtk.vtkButterflySubdivisionFilter()
#    subdivision.SetInputConnection(delaunay.GetOutputPort())
#    subdivision.Update()

    #smoother = vtk.vtkWindowedSincPolyDataFilter()
    #smoother.SetInputConnection(delaunay.GetOutputPort())
    #smoother.SetNumberOfIterations(5)
    #smoother.BoundarySmoothingOff()
    #smoother.FeatureEdgeSmoothingOff()
    #smoother.SetFeatureAngle(120.0)
    #smoother.SetPassBand(.001)
    #smoother.NonManifoldSmoothingOff()
    #smoother.NormalizeCoordinatesOff()
    #smoother.Update()

    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(delaunay.GetOutput())
    appendFilter.Update()

    unstructuredGrid = vtk.vtkUnstructuredGrid()
    unstructuredGrid.ShallowCopy(appendFilter.GetOutput())

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('cosipy.vtu')
    writer.SetInputData(unstructuredGrid)
    writer.Write()

def createDEM_v2():

    ds = xr.open_dataset('../data/output/Peru_20160601-20180530_comp4.nc')
    
    points = vtk.vtkPoints()
    quad = vtk.vtkQuad()
    cells = vtk.vtkCellArray()
    
    numPoints = ds.south_north.size*ds.west_east.size
   
    print('Write points \n')
    for i,j in product(ds.south_north.values,ds.west_east.values):
            points.InsertNextPoint(ds.lat.isel(south_north=i,west_east=j), ds.lon.isel(south_north=i,west_east=j), ds.HGT.sel(south_north=i,west_east=j).values/6370000.0)
    
    print('Write cells \n')
    for idx in range(points.GetNumberOfPoints()-ds.west_east.size):
        if (idx%ds.west_east.size != 0):
            quad.GetPointIds().SetId(0,idx)
            quad.GetPointIds().SetId(1,idx+1)
            quad.GetPointIds().SetId(2,idx+ds.west_east.size+1)
            quad.GetPointIds().SetId(3,idx+ds.west_east.size)
            cells.InsertNextCell(quad)

    print('Create unstructured grid \n') 
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(vtk.VTK_QUAD, cells)
    
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('cosipy.vtu')
    writer.SetInputData(grid)
    writer.Write()
   
def add_scalar(var, timestamp):

    vtkFile = vtk.vtkXMLUnstructuredGridReader()
    vtkFile.SetFileName('cosipy.vtu')
    vtkFile.Update()
    
    # Find cellId by coordinates
    pointLocator =  vtk.vtkPointLocator()
    pointLocator.SetDataSet(vtkFile.GetOutput())
    pointLocator.BuildLocator()
    
    ds = xr.open_dataset('../data/output/Peru_20160601-20180530_comp4.nc')
    ds = ds.sel(time=timestamp)
    
    ds_sub = ds[var].stack(x=['south_north','west_east']) 
    ds_sub = ds_sub.dropna(dim='x')
    lats = ds_sub.x.lat.values
    lons = ds_sub.x.lon.values
    data = ds_sub.values
    print(lats)

    numPoints = vtkFile.GetOutput().GetNumberOfPoints()
    scalar = np.empty(numPoints)
    scalar[:] = np.nan

    interpField = numpy_support.numpy_to_vtk(scalar)
    interpField.SetName(var)
    vtkFile.GetOutput().GetPointData().AddArray(interpField)
    vtkFile.Update()

    print('Write points \n')
    for i in np.arange(len(data)):
        # Get height
        alt = ds.HGT.sel(lat=lats[i],lon=lons[i]).values/6370000.0
        
        pointId = vtk.mutable(0) 
        Id = vtk.vtkIdList()
        pointId = pointLocator.FindClosestPoint([lons[i],lats[i],alt])
        vtkFile.GetOutput().GetPointData().GetArray(var).InsertTuple1(pointId,data[i])

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName('cosipy.vtu')
    writer.SetInputData(vtkFile.GetOutput())
    writer.Write()

    #plotSurface(vtkFile)

    
def plotSurface(domain):

    print(domain)
    domain.GetOutput().GetPointData().SetActiveScalars('TS')
    
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(16)
    lut.SetHueRange(2/3,0.0)
    #lut.SetValueRange (270, 273);
    lut.SetNanColor(1,1,1,0.5)
    lut.SetNumberOfTableValues(3)
    lut.SetNumberOfColors(16)
    lut.Build()

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    WIDTH=1200
    HEIGHT=1200
    renWin.SetSize(WIDTH,HEIGHT)
     
    # create a renderwindowinteractor
    joystickStyle = vtk.vtkInteractorStyleJoystickCamera()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(joystickStyle)

    # mapper
    coneMapper = vtk.vtkDataSetMapper()
    coneMapper.SetInputData(domain.GetOutput())
    coneMapper.SetScalarModeToUsePointData()
    coneMapper.ScalarVisibilityOn()
    coneMapper.SetLookupTable(lut)
    coneMapper.SetScalarRange(domain.GetOutput().GetPointData().GetArray("TS").GetRange())

    # actor
    coneActor = vtk.vtkActor()
    coneActor.SetMapper(coneMapper)
    coneActor.SetScale(1,1,50)
    coneActor.GetProperty().SetEdgeColor(0.7,0.7,0.7)
    #coneActor.GetProperty().EdgeVisibilityOn()
    coneActor.GetProperty().SetAmbient(0.0)
    coneActor.GetProperty().SetSpecularPower(0)
    coneActor.GetProperty().SetDiffuse(1)

    cam = vtk.vtkCamera()
    cam.SetPosition(10.89,46.83,0.07)
    #cam.SetPosition(10.89,46.83,0.08)
    cam.SetFocalPoint(10.82,46.83,0.02)
    cam.SetViewUp(0,0,1)

    # assign actor to the renderer
    ren.SetBackground(1,1,1)
    ren.AddActor(coneActor)
    ren.SetActiveCamera(cam)
    ren.ResetCamera()

    # enable user interface interactor
    #iren.Initialize()
    renWin.OffScreenRenderingOn()
    renWin.Render()
    #iren.Start()
   
    renderLarge = vtk.vtkRenderLargeImage()
    renderLarge.SetInput(ren)
    renderLarge.SetMagnification(5)

#    w2if = vtk.vtkWindowToImageFilter()
#    w2if.SetInputConnection(renderLarge.GetOutputPort())
#    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(renderLarge.GetOutputPort())
    writer.SetFileName('cosipy.png')
    writer.Write()


if __name__ == '__main__':
    main()








