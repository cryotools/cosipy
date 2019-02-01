#!/Users/tsauter/anaconda/bin/python
#coding:utf-8
# Purpose: Export 3D objects, build of faces with 3 or 4 vertices, as ASCII or Binary STL file.
# License: MIT License

import struct

from osgeo.gdalconst import *  
from osgeo import gdal 
from osgeo import ogr
import numpy as np
import optparse

ASCII_FACET = """facet normal 0 0 0
outer loop
vertex {face[0][0]:.4f} {face[0][1]:.4f} {face[0][2]:.4f}
vertex {face[1][0]:.4f} {face[1][1]:.4f} {face[1][2]:.4f}
vertex {face[2][0]:.4f} {face[2][1]:.4f} {face[2][2]:.4f}
endloop
endfacet
"""

BINARY_HEADER ="80sI"
BINARY_FACET = "12fH"

class ASCII_STL_Writer:
    """ Export 3D objects build of 3 or 4 vertices as ASCII STL file.
    """
    def __init__(self, stream):
        self.fp = stream
        self._write_header()

    def _write_header(self):
        self.fp.write("solid python\n")

    def close(self):
        self.fp.write("endsolid python\n")

    def _write(self, face):
        self.fp.write(ASCII_FACET.format(face=face))

    def _split(self, face):
        p1, p2, p3, p4 = face
        return (p1, p2, p3), (p3, p4, p1)

    def add_face(self, face):
        """ Add one face with 3 or 4 vertices. """
        if len(face) == 4:
            face1, face2 = self._split(face)
            self._write(face1)
            self._write(face2)
        elif len(face) == 3:
            self._write(face)
        else:
            raise ValueError('only 3 or 4 vertices for each face')

    def add_faces(self, faces):
        """ Add many faces. """
        for face in faces:
            self.add_face(face)

class Binary_STL_Writer(ASCII_STL_Writer):
    """ Export 3D objects build of 3 or 4 vertices as binary STL file.
    """
    def __init__(self, stream):
        self.counter = 0
        super(Binary_STL_Writer, self).__init__(stream)

    def close(self):
        self._write_header()

    def _write_header(self):
        self.fp.seek(0)
        self.fp.write(struct.pack(BINARY_HEADER, b'Python Binary STL Writer', self.counter))

    def _write(self, face):
        self.counter += 1
        data = [
            0., 0., 0.,
            face[0][0], face[0][1], face[0][2],
            face[1][0], face[1][1], face[1][2],
            face[2][0], face[2][1], face[2][2],
            0
        ]
        self.fp.write(struct.pack(BINARY_FACET, *data))
    

def example(DEMfile,maskFile):

    #---------------------------------------------------
    # Read digital elevation model from file and create
    # plot
    #---------------------------------------------------
    def read_dem(DEMfile, maskFile):

    	print "-------------------------------"
    	print "Reading entire domain"
    	print "-------------------------------\n"
    	print "... %s" % DEMfile
    
    	# Opening the raster file  
    	dataset = gdal.Open(DEMfile, GA_ReadOnly )  
    	band = dataset.GetRasterBand(1)  
    	
    	# Reading the raster properties  
    	projectionfrom = dataset.GetProjection() 
    	geotransform = dataset.GetGeoTransform()  
    	xsize = band.XSize  
    	ysize = band.YSize  
    	datatype = band.DataType  
    			
    	# Reading the raster values  
    	values = band.ReadAsArray()  
    	values[values == 32767] = 0
    	values[values < 0] = 0
    	
    	# Get extent of the DEM
    	extent = (geotransform[0], geotransform[0] + \
    			dataset.RasterXSize * geotransform[1], \
    			geotransform[3] + dataset.RasterYSize * \
    			geotransform[5], geotransform[3])
    	
    	print "-------------------------------"
    	print "Reading mask"
    	print "-------------------------------\n"
    	print "... %s" % maskFile
    
    	# Opening the raster file  
    	mask = gdal.Open(maskFile, GA_ReadOnly )  
    	bandMask = mask.GetRasterBand(1)  
    	
    	# Reading the raster properties  
    	projectionfromMask = mask.GetProjection() 
    	geotransformMask = mask.GetGeoTransform()  
    	xsizeMask = bandMask.XSize  
    	ysizeMask = bandMask.YSize  
    	datatypeMask = bandMask.DataType  
    			
    	# Reading the raster values  
    	valuesMask = bandMask.ReadAsArray()  
    	valuesMask[valuesMask == 32767] = 0
    	valuesMask[valuesMask < 0] = 0
    	
    	# Get extent of the DEM
    	extentMask = (geotransformMask[0], geotransformMask[0] + \
    			mask.RasterXSize * geotransformMask[1], \
    			geotransformMask[3] + mask.RasterYSize * \
    			geotransformMask[5], geotransformMask[3])

    	print "... DONE"
    	print "-------------------------------\n"

        ptDomain = []
        ptMask = []

        for y in range(0,ysize-1):
            for x in range(0,xsize-1):
                
                p1 = ((x*geotransform[1], ((ysize-1)*abs(geotransform[5]))+
			y*geotransform[5], values[y,x]))
                p2 = (((x+1)*geotransform[1], ((ysize-1)*abs(geotransform[5]))+
			y*geotransform[5], values[y,(x+1)]))
                p3 = (((x+1)*geotransform[1], ((ysize-1)*abs(geotransform[5]))+
			(y+1)*geotransform[5], values[(y+1),(x+1)]))
                p4 = ((x*geotransform[1], ((ysize-1)*abs(geotransform[5]))+
			(y+1)*geotransform[5], values[(y+1),x]))
               
                if not (valuesMask[y,x] == 0):
                    ptMask.append([p1,p2,p3,p4])
                else:
                    ptDomain.append([p1,p2,p3,p4])

	print (xsize-1)*geotransform[1]
	print (ysize-1)*geotransform[5]

        return (ptDomain, ptMask)
    

    def get_cube():
        # cube corner points
        s = 3.
        p1 = (0, 0, 0)
        p2 = (0, 0, s)
        p3 = (0, s, 0)
        p4 = (0, s, s)
        p5 = (s, 0, 0)
        p6 = (s, 0, s)
        p7 = (s, s, 0)
        p8 = (s, s, s)

        # define the 6 cube faces
        # faces just lists of 3 or 4 vertices
        return [
            [p1, p5, p7, p3],
            [p1, p5, p6, p2],
            [p5, p7, p8, p6],
            [p7, p8, p4, p3],
            [p1, p3, p4, p2],
            [p2, p6, p8, p4],
        ]

    subset = read_dem(DEMfile,maskFile) 
    domain = subset[0]
    mask = subset[1]
    
    with open('domain.stl', 'wb') as fp:
        writer = ASCII_STL_Writer(fp)
        writer.add_faces(domain)
        writer.close()

    with open('mask.stl', 'wb') as fp:
        writer = ASCII_STL_Writer(fp)
        writer.add_faces(mask)
        writer.close()




if __name__ == '__main__':

	parser = optparse.OptionParser("createPatch -d <DEM file> -s <mask file>")
	parser.add_option('-d', '--DEMfile',
	                  dest="DEMfile",
	                  default=False,
	                  action="store",
			  help="Digital Elevation Model",
	                  )
	parser.add_option('-s',
	                  dest="maskFile",
	                  default=False,
	                  action="store",
			  help="Mask file",
	                  )
	(option, args) = parser.parse_args()
    	
	example(option.DEMfile,option.maskFile)
