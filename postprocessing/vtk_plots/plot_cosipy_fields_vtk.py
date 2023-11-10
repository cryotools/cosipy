import argparse
import re
import warnings
from itertools import product

import numpy as np
import vtk
import xarray as xr
from vtk.util import numpy_support


def main():
    args = parse_arguments()

    createDEM_v1(file_path=args.input_file)
    # createDEM_v2(file_path=args.input_file)
    add_scalar(
        file_path=args.input_file,
        var=args.name,
        timestamp=args.timestamp,
        mean=args.mean,
        gltf=args.gltf,
    )


def check_2d(array: xr.DataArray):
    """Checks if an input array has 2D spatial coordinates.

    Raises:
        ValueError: Spatial coordinates are not 2D.
    """

    for dimension in array.dims:
        if dimension not in ["time", "layer"] and (array.dims[dimension]) <= 1:
            raise ValueError("Spatial coordinates are not 2D.")


def get_selection(
    array: xr.Dataset, timestamp: str, mean: bool = False
) -> xr.DataArray:
    """Selects array from dataset at specific time or as a daily mean.

    Args:
        array (xr.Dataset or xr.DataArray): Labelled data.
        timestamp: Time index of target data.
        mean: If True, computes and selects the daily mean. Otherwise,
            selects data at ``timestamp``. Default False.

    Returns:
        Array selection at target time.
    """

    if not mean:
        data = array.sel(time=timestamp, method="nearest")
    else:
        data = (
            array.resample(time="1D", skipna=True)
            .mean()
            .sel(time=timestamp, method="nearest")
        )

    return data


def get_coords(data) -> tuple:
    """Get latitude/longitude keys.

    Args:
        data (xr.Dataset or xr.DataArray): Labelled dataset.

    Returns:
        tuple[str, str]: Keys referring to latitude and longitude.
    """

    if "south_north" in data.keys():
        northing = "south_north"
        easting = "west_east"
    else:
        northing = "lat"
        easting = "lon"

    return northing, easting


def create_points(data: xr.Dataset) -> vtk.vtkPoints:
    """Convert xarray dataset to points."""

    print("Writing points...\n")
    latitude, longitude = get_coords(data)
    points = vtk.vtkPoints()
    if "south_north" in data.keys():
        for i, j in product(data[longitude].values, data[latitude].values):
            points.InsertNextPoint(
                data[longitude].sel(west_east=i),
                data[latitude].sel(south_north=j),
                data.HGT.sel(west_east=i, south_north=j).values / 6370000.0,
            )
    else:
        for i, j in product(data[longitude].values, data[latitude].values):
            points.InsertNextPoint(
                data[longitude].sel(lon=i),
                data[latitude].sel(lat=j),
                data.HGT.sel(lon=i, lat=j).values / 6370000.0,
            )

    return points


def create_unstructured_grid(
    points: vtk.vtkPoints, cells: vtk.vtkCellArray = None
) -> vtk.vtkUnstructuredGrid:
    """Creates unstructured grid from points and/or cells."""

    print("Create unstructured grid...\n")
    grid = vtk.vtkUnstructuredGrid()
    if points is not None:
        grid.SetPoints(points)
    if cells is not None:
        grid.SetCells(vtk.VTK_QUAD, cells)

    return grid


def write_unstructured_grid_to_file(
    unstructured_grid: vtk.vtkUnstructuredGrid, file_name: str = "cosipy.vtu"
):
    """Writes unstructured grid data to .vtu file."""

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(unstructured_grid)
    writer.Write()


def write_render_to_gltf(
    render: vtk.vtkRenderer,
    variable: str,
    timestamp: str,
):
    """Writes rendered image to a 3D .gltf object.

    Raises:
        AttributeError: GLTF export is only supported in VTK >9.2.
    """

    print("Saving 3D object...\n")
    output_file = set_filename(name=variable, timestamp=timestamp, fmt="gltf")
    try:
        writer = vtk.vtkGLTFExporter()
        writer.InlineDataOn()
        writer.SetActiveRenderer(render)
        writer.SetRenderWindow(render.GetRenderWindow())
        writer.SetFileName(output_file)
        writer.Update()
        writer.Write()
        print("3D object saved.\n")
    except AttributeError:
        print("GLTF export is only supported in VTK >9.2.")


def write_render_to_image(
    render: vtk.vtkRenderLargeImage, variable: str, timestamp: str
):
    """Writes rendered image to .png file."""

    print("Saving image...\n")
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(render.GetOutputPort())
    output_file = set_filename(name=variable, timestamp=timestamp)
    writer.SetFileName(output_file)
    writer.Write()
    print("Image saved.\n")


def smooth_mesh(mesh: vtk.vtkPolyData) -> vtk.vtkButterflySubdivisionFilter:
    """Smooths mesh using butterfly subdivision."""

    subdivision = vtk.vtkButterflySubdivisionFilter()
    subdivision.SetInputData(mesh)
    subdivision.Update()

    # smoother = vtk.vtkWindowedSincPolyDataFilter()
    # smoother.SetInputConnection(delaunay.GetOutputPort())
    # smoother.SetNumberOfIterations(5)
    # smoother.BoundarySmoothingOff()
    # smoother.FeatureEdgeSmoothingOff()
    # smoother.SetFeatureAngle(120.0)
    # smoother.SetPassBand(0.001)
    # smoother.NonManifoldSmoothingOff()
    # smoother.NormalizeCoordinatesOff()
    # smoother.Update()

    return subdivision


def createDEM_v1(
    file_path: str = "../../data/output/Zhadang_ERA5_20090101-20090110.nc",
):
    """Map dataset to DEM using Delaunay triangulation.

    The DEM is saved as a vtu file.

    .. todo:: add terrain smoothing.

    .. note:: smoothing the DEM on creation means add_scalar will not
        find the closest points, as the planes no longer intersect. If
        smoothing occurs at the start of plotSurface, the data lost
        through triangulating is not recovered.
    """

    ds = xr.open_dataset(file_path)
    check_2d(ds)
    print("Creating DEM...\n")

    points = create_points(data=ds)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(polydata)
    delaunay.Update()

    append_filter = vtk.vtkAppendFilter()
    append_filter.AddInputData(delaunay.GetOutput())
    append_filter.Update()

    grid = vtk.vtkUnstructuredGrid()
    grid.ShallowCopy(append_filter.GetOutput())
    write_unstructured_grid_to_file(grid, "cosipy.vtu")


def createDEM_v2(
    file_path: str = "../../data/output/Zhadang_ERA5_20090101-20090110.nc",
):
    """Map dataset to DEM through indexing.

    .. note:: This method causes graphical glitches. Repeating
        quad.GetPointIds for longitude gives a more accurate shape, but
        the contours do not interpolate correctly so the colourmap
        values are inaccurate.
    """

    ds = xr.open_dataset(file_path)
    check_2d(ds)
    print("Creating DEM...\n")
    points = create_points(data=ds)

    print("Writing cells...\n")
    quad = vtk.vtkQuad()
    cells = vtk.vtkCellArray()
    for idx in range(points.GetNumberOfPoints() - ds.lat.size):
        if idx % ds.lat.size != 0:
            quad.GetPointIds().SetId(0, idx)
            quad.GetPointIds().SetId(1, idx + 1)
            quad.GetPointIds().SetId(2, idx + ds.lat.size + 1)
            quad.GetPointIds().SetId(3, idx + ds.lat.size)
            cells.InsertNextCell(quad)

    grid = create_unstructured_grid(points=points, cells=cells)
    write_unstructured_grid_to_file(grid, "cosipy.vtu")


def add_scalar(
    file_path: str,
    var: str,
    timestamp: str,
    mean: bool = False,
    gltf: bool = False,
):
    """Selects data from array at specific time or as a daily mean.

    .. todo:: support WRF coordinates (south_north/west_east)

    Args:
        file_path: Path to netcdf file.
        var: Short name of variable in data.
        timestamp: Time index of target data.
        mean: If True, computes and selects the daily mean. Otherwise,
            selects data at ``timestamp``. Default False.
        gltf: If True, export as a 3D glTF object. Default False.

    Returns:
        Array selection at target time.
    """

    vtk_file = vtk.vtkXMLUnstructuredGridReader()
    vtk_file.SetFileName("cosipy.vtu")
    vtk_file.Update()

    # Find cellId by coordinates
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(vtk_file.GetOutput())
    point_locator.BuildLocator()

    ds = xr.open_dataset(file_path)
    ds = get_selection(array=ds, timestamp=timestamp, mean=mean)

    latitude, longitude = get_coords(data=ds)
    ds_sub = ds[var].stack(x=[latitude, longitude])
    ds_sub = ds_sub.dropna(dim="x")
    lats = ds_sub[latitude].values
    lons = ds_sub[longitude].values
    data = ds_sub.values

    num_points = vtk_file.GetOutput().GetNumberOfPoints()
    scalar = np.empty(num_points)
    scalar[:] = np.nan

    interp_field = numpy_support.numpy_to_vtk(scalar)
    interp_field.SetName(var)
    vtk_file.GetOutput().GetPointData().AddArray(interp_field)
    vtk_file.Update()

    print("Writing points...\n")
    for i in np.arange(len(data)):
        # Get height
        alt = ds.HGT.sel(lat=lats[i], lon=lons[i]).values / 6370000.0

        point_id = vtk.mutable(0.0)
        point_id = point_locator.FindClosestPoint([lons[i], lats[i], alt])
        if point_id < 0:
            warnings.warn("tupleIndex from pointId < 0.")
        else:
            vtk_file.GetOutput().GetPointData().GetArray(var).InsertTuple1(
                point_id, data[i]
            )

    write_unstructured_grid_to_file(vtk_file.GetOutput(), "cosipy.vtu")

    plotSurface(domain=vtk_file, variable=var, timestamp=timestamp, gltf=gltf)


def convert_unstructured_to_polydata(
    unstructured_grid: vtk.vtkUnstructuredGrid,
) -> vtk.vtkPolyData:
    """Converts unstructured grid to polydata.

    More complex filters only accept polydata.
    """

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_grid)
    geometry_filter.Update()
    polydata = geometry_filter.GetOutput()
    msg = f"Converted to polydata with {polydata.GetNumberOfPoints()} points."
    print(msg)

    return polydata


def set_filename(name: str, timestamp: str, fmt: str = "png") -> str:
    """Creates a file name from timestamp and variable name.

    Args:
        name: Name of data variable.
        timestamp: Time of data collection.
        fmt: File format. Default "png".

    Returns:
        Output file name.
    """

    if not isinstance(timestamp, str):
        img_id = timestamp.strftime("%Y%m%d")
    else:
        img_id = timestamp
    img_id = re.sub(r"\W+", "", str(img_id))  # avoid illegal file names
    img_id = f"{img_id}_{name}_vtk.{fmt}"

    return img_id


def plotSurface(
    domain: vtk.vtkXMLUnstructuredGridReader,
    variable: str,
    timestamp: str = "",
    contours: int = 10,
    gltf: bool = False,
) -> vtk.vtkRenderLargeImage:
    """Plot surface with contoured data.

    Args:
        domain: Reader for glacier surface XML data.
        variable: Short name of target variable.
        timestamp: Time index of target data. Default empty string.
        contours: Number of contours to plot. Default 10.
        gltf: Export as 3D glTF object. Default False.

    Returns:
        Large image render of plotted data.
    """

    print(domain)
    domain.GetOutput().GetPointData().SetActiveScalars(variable)
    scalar_range = domain.GetOutput().GetPointData().GetArray(variable).GetRange()
    num_contours = contours

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(num_contours + 1)
    if variable.lower() == "hgt":
        lut.SetHueRange(0.75, 0.1)  # blue: low altitude
    else:
        lut.SetHueRange(0.1, 0.75)  # matches cmap in plot_cosipy_profiles
    lut.SetNanColor(1, 1, 1, 0.5)
    lut.Build()

    # mapper
    cone_mapper = vtk.vtkDataSetMapper()
    cone_mapper.SetInputData(
        convert_unstructured_to_polydata(domain.GetOutput())  # for 3D export
    )
    cone_mapper.SetScalarModeToUsePointData()
    cone_mapper.ScalarVisibilityOn()
    cone_mapper.SetLookupTable(lut)
    cone_mapper.SetScalarRange(scalar_range[0], scalar_range[1])
    cone_mapper.SetInterpolateScalarsBeforeMapping(1)  # stops colour smudging
    cone_mapper.Update()

    # actor
    cone_actor = vtk.vtkActor()
    cone_actor.SetMapper(cone_mapper)
    cone_actor.SetScale(1, 1, 50)
    cone_actor.GetProperty().SetEdgeColor(0.7, 0.7, 0.7)
    cone_actor.GetProperty().EdgeVisibilityOn()
    cone_actor.GetProperty().SetAmbient(0.0)
    cone_actor.GetProperty().SetSpecularPower(0)
    cone_actor.GetProperty().SetDiffuse(1)
    cone_actor.GetProperty().SetInterpolationToFlat()  # stop contour blending

    # visualisation
    camera = vtk.vtkCamera()
    camera.SetPosition(0, 0, 1)  # x: west-east, y: south-north, z: top-down
    camera.SetFocalPoint(0, 1, 0)
    # cam.SetPosition(10.89, 46.83, 0.07)
    # cam.SetPosition(10.89,46.83,0.08)
    # cam.SetFocalPoint(10.82, 46.83, 0.02)
    # cam.SetViewUp(0, 0, 1)

    # create a rendering window and renderer
    render = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(render)
    width = 1200
    height = 1200
    render_window.SetSize(width, height)

    # create interactor for render window
    joystick_style = vtk.vtkInteractorStyleJoystickCamera()
    interactive_render = vtk.vtkRenderWindowInteractor()
    interactive_render.SetRenderWindow(render_window)
    interactive_render.SetInteractorStyle(joystick_style)
    # assign actor to renderer
    render.SetBackground(1, 1, 1)
    render.AddActor(cone_actor)
    render.SetActiveCamera(camera)
    render.ResetCamera()

    # enable user interface interactor
    # interactive_render.Initialize()
    render_window.OffScreenRenderingOn()
    render_window.Render()
    # interactive_render.Start()

    render_large = vtk.vtkRenderLargeImage()
    render_large.SetInput(render)
    render_large.SetMagnification(5)

    #    w2if = vtk.vtkWindowToImageFilter()
    #    w2if.SetInputConnection(render_large.GetOutputPort())
    #    w2if.Update()

    write_render_to_image(render=render_large, variable=variable, timestamp=timestamp)

    if gltf:
        write_render_to_gltf(render=render, variable=variable, timestamp=timestamp)

    return render_large


def parse_arguments() -> argparse.Namespace:
    """Parse user arguments.

    Required arguments:
        -i, --input <path>      Path to .nc file.
        -t, --time <str>        Target date or timestamp.
        -n, --name <str>        Name of variable to plot as contours.

    Optional switches:
        -h, --help              Show this help message and exit.
        -m, --mean              Plot daily mean instead of timestep.
                                    Default False.
        -g, --gltf              Export plot as a 3D glTF object.
                                    Default False.
    """

    tagline = "Plot 3D DEM with contours for a single variable."
    parser = argparse.ArgumentParser(
        prog="plot_cosipy_fields_vtk.py", description=tagline
    )
    # Required
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        required=True,
        default=None,
        type=str,
        metavar="<path>",
        help="Path to .nc file",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        type=str,
        default=None,
        metavar="<str>",
        help="Name of variable to plot as contours.",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timestamp",
        type=str,
        required=True,
        default=None,
        metavar="<str>",
        help="Target date or timestamp",
    )

    # Optional switches
    parser.add_argument(
        "-m",
        "--mean",
        dest="mean",
        action="store_true",
        help="Plot daily mean instead of timestep",
    )
    parser.add_argument(
        "-g",
        "--gltf",
        dest="gltf",
        action="store_true",
        help="Export plot as a 3D glTF object",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    main()
