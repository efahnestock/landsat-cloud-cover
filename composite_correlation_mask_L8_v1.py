import os, sys, glob
from matplotlib import pyplot as plt
#### IMPORTS FOR IMPORTING IMAGES #####
import numpy as np
import datetime as dt
import gdal, osr
import argparse


#################################################################
# declare classes and functions up front so they are easy to find
#################################################################

class GeoImgNoDir:  # modified 9/7/2015 for LO8 fname -> date    modified to noload for sc application - don't read image on setup - will read image data in main code to hp filter, delete...and keep memory footprint small
	"""geocoded image input and info
		a=GeoImg(in_file_name)
			a.img will contain image
			a.parameter etc..."""
	def __init__(self, in_filename, datestr=None,datefmt='%m/%d/%y'):
		self.filename = in_filename
#                 self.in_dir_path = in_dir  #in_dir can be relative...
#                 self.in_dir_abs_path=os.path.abspath(in_dir)  # get absolute path for later ref if needed
		self.gd=gdal.Open(self.filename)
		self.nodata_value=self.gd.GetRasterBand(1).GetNoDataValue()
		self.srs=osr.SpatialReference(wkt=self.gd.GetProjection())
		self.gt=self.gd.GetGeoTransform()
		self.proj=self.gd.GetProjection()
		self.intype=self.gd.GetDriver().ShortName
		self.min_x=self.gt[0]
		self.max_x=self.gt[0]+self.gd.RasterXSize*self.gt[1]
		self.min_y=self.gt[3]+self.gt[5]*self.gd.RasterYSize
		self.max_y=self.gt[3]
		self.pix_x_m=self.gt[1]
		self.pix_y_m=self.gt[5]
		self.num_pix_x=self.gd.RasterXSize
		self.num_pix_y=self.gd.RasterYSize
		self.XYtfm=np.array([self.min_x,self.max_y,self.pix_x_m,self.pix_y_m]).astype('float')
		if (datestr is not None):
			self.imagedatetime=dt.datetime.strptime(datestr,datefmt)
		elif ((self.filename.find('LC8') == 0) | (self.filename.find('LO8') == 0) | (self.filename.find('LE7') == 0) | (self.filename.find('LT5') == 0) | (self.filename.find('LT4') == 0)):      # looks landsat like - try parsing the date from filename (contains day of year)
			self.sensor=self.filename[0:3]
			self.path=int(self.filename[3:6])
			self.row=int(self.filename[6:9])
			self.year=int(self.filename[9:13])
			self.doy=int(self.filename[13:16])
			self.imagedatetime=dt.datetime.fromordinal(dt.date(self.year-1,12,31).toordinal()+self.doy)
		else:
			self.imagedatetime=None  # need to throw error in this case...or get it from metadata
		self.img=self.gd.ReadAsArray().astype(np.float32)   # works for L8 and earlier - and openCV correlation routine needs float or byte so just use float...
		self.img_ov2=self.img[0::2,0::2]
		self.img_ov10=self.img[0::10,0::10]
	def imageij2XY(self,ai,aj,outx=None,outy=None):
		it = np.nditer([ai,aj,outx,outy],
						flags = ['external_loop', 'buffered'],
						op_flags = [['readonly'],['readonly'],
									['writeonly', 'allocate', 'no_broadcast'],
									['writeonly', 'allocate', 'no_broadcast']])
		for ii,jj,ox,oy in it:
			ox[...]=(self.XYtfm[0]+((ii+0.5)*self.XYtfm[2]));
			oy[...]=(self.XYtfm[1]+((jj+0.5)*self.XYtfm[3]));
		return np.array(it.operands[2:4])
	def XY2imageij(self,ax,ay,outi=None,outj=None):
		it = np.nditer([ax,ay,outi,outj],
						flags = ['external_loop', 'buffered'],
						op_flags = [['readonly'],['readonly'],
									['writeonly', 'allocate', 'no_broadcast'],
									['writeonly', 'allocate', 'no_broadcast']])
		for xx,yy,oi,oj in it:
			oi[...]=((xx-self.XYtfm[0])/self.XYtfm[2])-0.5;  # if python arrays started at 1, + 0.5
			oj[...]=((yy-self.XYtfm[1])/self.XYtfm[3])-0.5;  # " " " " "
		return np.array(it.operands[2:4])
		
		
		
		
# set up command line arguments
parser = argparse.ArgumentParser( \
    description="""build composite (max) correlation image from all velocity pairs the input landsat image was used in - for cloud masking that image.
    """,
#     epilog='>>  <<',
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-band8_dir', 
                    action='store', 
                    type=str, 
                    default='.',
                    help='top band 8 directory - directory to find p???_r??? directory containing specified band 8 image in [%(default)s]')
parser.add_argument('input_band8_image', 
                    action='store', 
                    type=str,
                    help='what it says')
parser.add_argument('-output_mask_pixel_size_m', 
                    action='store', 
                    type=float,
                    default=300.0,
                    help='pixel size in meters for mask [%(default)6.2f]')
parser.add_argument('-nodataval', 
                    action='store', 
                    type=float,
                    default=-2.0,
                    help='nodatavalue for output mask [%(default)f]')
parser.add_argument('-velocity_data_dir', 
                    action='store', 
                    type=str, 
                    default='.',
                    help='top directory to find p???_r??? directories containing velocity .nc files in [%(default)s]')
parser.add_argument('-output_dir', 
                    action='store', 
                    type=str,
                    default='.', 
                    help='directory to put max correlation output geotiff in [%(default)s]')
parser.add_argument('-output_name_tail', 
                    action='store', 
                    type=str,
                    default='_max_corr_array', 
                    help='end of name for max correlation output geotiff [%(default)s]')
# next argument is just a flag - when specified on the command line, it will be True
parser.add_argument('-show_fig', 
                    action='store_true',
                    default=False, 
                    help='show output figure - [False]')
args = parser.parse_args()

# 
# os.chdir("/Users/aknh9189/Downloads/Landsat_front_tracing/L8_B8/p233_r016/")
# 

# infiles=glob.glob('L*.TIF') #glob da files
# in_prs=[(x[3:6],x[6:9],x[9:13],x[13:16]) for x in infiles] #get da data from da files



in_b8_file = args.input_band8_image
p,r,y,d = (in_b8_file[3:6],in_b8_file[6:9],in_b8_file[9:13],in_b8_file[13:16]) # p,r,y,d

in_b8_dir = args.band8_dir + '/p%s_r%s'%(p,r)



# in_prs=[(x[3:6],x[6:9],x[9:13],x[13:16]) for x in infiles] #get da data from da files

# in_vel_data_items = {} #stores all of the collected correleated files with key format yearday



in_vel_data=glob.glob(args.velocity_data_dir + '/p%s_r%s/*%s_%s*.nc'%(p,r,y,d)) #find every output that included this input filename


# [in_vel_data.append(x) for x in glob.glob('../../L8_out/p%s_r%s/grn_v00_S8_%s_%s_*_*_*_%s_%s_hp.nc'%(p,r,p,r,y,d))] #find every output where it is second
# def open_related_images(y,d):

nodataval = args.nodataval

input_values_found = False  # don't want to display output array or make geotiff of it if there wasn't anything to write...

if len(in_vel_data) != 0: #if there was data..
	# open the original band 8, make array to hold the mask
	orig_img = GeoImgNoDir(in_b8_dir + '/' + in_b8_file)
	corr_fields_max_vals = nodataval * np.ones((int((orig_img.max_y-orig_img.min_y)/args.output_mask_pixel_size_m), int((orig_img.max_x-orig_img.min_x)/args.output_mask_pixel_size_m))) #generate empty array to store the corr values in w/ dimientions
	center_pixel_x_vector = [orig_img.min_x + (args.output_mask_pixel_size_m/2.0) + (args.output_mask_pixel_size_m * i)  for i in range(0,int((orig_img.max_x-orig_img.min_x)/args.output_mask_pixel_size_m))]
	center_pixel_y_vector = [orig_img.max_y - ((args.output_mask_pixel_size_m/2.0) + (args.output_mask_pixel_size_m * j)) for j in range(0,int((orig_img.max_y-orig_img.min_y)/args.output_mask_pixel_size_m))] # note y decreases downward

	grid_thing_x, grid_thing_y = np.meshgrid(center_pixel_x_vector,center_pixel_y_vector) #creates the grid thing which is a list containing the x and y values

	# now find and open all the velocity .nc files (correlation arrays) that included this band 8 image
	for veldata_image in in_vel_data:
		full_gdal_name='NETCDF:\"' + veldata_image + '\":' + 'corr'
		corr_image = GeoImgNoDir(full_gdal_name)
	
		pointsij = np.where((grid_thing_x > corr_image.min_x) & (grid_thing_x < corr_image.max_x) & (grid_thing_y > corr_image.min_y) & (grid_thing_y < corr_image.max_y))
		if len(pointsij[0])>0:
			input_values_found = True
			#print len(pointsij[0]),corr_image.min_x,corr_image.min_y,corr_image.max_x,corr_image.max_y
			ptsx=grid_thing_x[pointsij] ; ptsy=grid_thing_y[pointsij]
			corr_array_sample_pts_i,corr_array_sample_pts_j=corr_image.XY2imageij(ptsx,ptsy)
			casp_rint_j=np.round(corr_array_sample_pts_j).astype('int')
			casp_rint_i=np.round(corr_array_sample_pts_i).astype('int')
			#print corr_image.img.shape, casp_rint_i.shape, casp_rint_j.shape
		
			image_corr_vals=corr_image.img[(casp_rint_j,casp_rint_i)]

			stack_corr_values=corr_fields_max_vals[pointsij]

			values_index = np.where((stack_corr_values<image_corr_vals))
			corr_fields_max_vals[(pointsij[0][values_index],pointsij[1][values_index])]=image_corr_vals[values_index]

			
if input_values_found:
	if args.show_fig:
		plt.figure()
		plt.imshow(corr_fields_max_vals, vmin=0.0, vmax=1.0)
		plt.colorbar()
		plt.title('%s: maximum correlation at each pixel'%(in_b8_file))
		plt.show()
	
	####################################	
	# now output corr field as a geotiff
	####################################
	format = "GTiff"
	driver = gdal.GetDriverByName( format )
	metadata = driver.GetMetadata()


	dst_filename = args.output_dir + '/' + in_b8_file.replace('.TIF','') + args.output_name_tail + '.tif'
	(out_lines,out_pixels)=corr_fields_max_vals.shape
	out_bands=1
	dst_ds = driver.Create(  dst_filename, out_pixels, out_lines, out_bands, gdal.GDT_Float32 )
	print 'out image %s %s %d, %d, %d'%(dst_filename, format, out_pixels, out_lines, out_bands)
	dst_ds.SetGeoTransform( [ orig_img.min_x, args.output_mask_pixel_size_m, 0, orig_img.max_y, 0, -args.output_mask_pixel_size_m ] ) # note pix_y_m typically negative (array stored top line first
	dst_ds.SetProjection( orig_img.proj )     # output image has same projection as original band 8 image, and same upper left corner, but a different pixel size
	dst_ds.GetRasterBand(1).SetNoDataValue( float( nodataval ) ) # add nodatavalue to gdal dataset
	dst_ds.GetRasterBand(1).WriteArray( (corr_fields_max_vals).astype('float32') )    # save as a 4-byte float, instead of numpy favored 8-byte float - smaller file
	dst_ds = None # done, close the dataset

else:
	print 'no overlapping correlation fields found?  are the directories wrong?'
	
	
	
# 	image_corr_fields = []
# 	"""Create GeoImgNoDir objects for all corr images with a common image. Year/day pair is entered to find the image"""
# 
# 
# 	try: #attempt to..
# 		for image in in_vel_data_items["%s%s"%(y,d)]: #get list of images for every input image
# 			image_corr_fields.append(GeoImgNoDir('NETCDF:"%s":corr'%(image))) #add all of the opened images to a dictonary with the same key
# 	except KeyError: # if there is nothing for that key
# 		pass
# 	return image_corr_fields
	

#orig_img = GeoImgNoDir("./LC82330162013102LGN01_B8.TIF")
## corr_fields = open_related_images('2013','102')
##print orig_img.min_x,orig_img.max_x,orig_img.min_y,orig_img.max_y
#
#corr_fields_max_vals = np.zeros((int((orig_img.max_x-orig_img.min_x)/300), int((orig_img.max_y-orig_img.min_y)/300))) #generate empty array to store the corr values in w/ dimientions
#center_pixel_x_vector = [150+300*x+orig_img.min_x for x in range(0,int(orig_img.max_x-orig_img.min_x)/300)]
#center_pixel_y_vector = [150+300*x+orig_img.min_y for x in range(0,int(orig_img.max_y-orig_img.min_y)/300)]
##print len(center_pixel_y_vector)
#
#grid_thing_x, grid_thing_y = np.meshgrid(center_pixel_x_vector,center_pixel_y_vector) #creates the grid thing which is a list containing the x and y values
#
#pointsij = np.where((grid_thing_x > image.min_x) & (grid_thing_x < image.max_x) &
#						 (grid_thing_y > image.min_y) & (grid_thing_y < image.max_y))
#
#ptsx=grid_thing_x[pointsij] ; ptsy=grid_thing_y[pointsij]
#
#corr_array_sample_pts_i,corr_array_sample_pts_j=image.XY2imageij(ptsx,ptsy)
#
#casp_rint_j=np.round(corr_array_sample_pts_j).astype('int')
#casp_rint_i=np.round(corr_array_sample_pts_i).astype('int')
# img_dt = "2015268"
# try:
# 	orig_img = GeoImgNoDir("./LC8233016"+img_dt+"LGN00_B8.TIF")
# except:
# 	orig_img = GeoImgNoDir("./LC8233016"+img_dt+"LGN01_B8.TIF")
# 
# corr_fields_max_vals = np.zeros((int((orig_img.max_x-orig_img.min_x)/300), int((orig_img.max_y-orig_img.min_y)/300))) #generate empty array to store the corr values in w/ dimientions
# center_pixel_x_vector = [150+300*x+orig_img.min_x for x in range(0,int(orig_img.max_x-orig_img.min_x)/300)]
# center_pixel_y_vector = [150+300*x+orig_img.min_y for x in range(0,int(orig_img.max_y-orig_img.min_y)/300)]
# 
# grid_thing_y, grid_thing_x = np.meshgrid(center_pixel_y_vector,center_pixel_x_vector) #creates the grid thing which is a list containing the x and y values
# 
# #print 'sizes:',corr_fields_max_vals.shape,grid_thing_x.shape
# 

# for corr_field_addr in in_vel_data_items[img_dt]: #for a specific image
# 	print corr_field_addr
# 	corr_image = GeoImgNoDir('NETCDF:"'+corr_field_addr+'":corr')
# 	pointsij = np.where((grid_thing_x > corr_image.min_x) & (grid_thing_x < corr_image.max_x) & (grid_thing_y > corr_image.min_y) & (grid_thing_y < corr_image.max_y))
# 	if len(pointsij[0])>0:
# 		#print len(pointsij[0]),corr_image.min_x,corr_image.min_y,corr_image.max_x,corr_image.max_y
# 		ptsx=grid_thing_x[pointsij] ; ptsy=grid_thing_y[pointsij]
# 		corr_array_sample_pts_i,corr_array_sample_pts_j=corr_image.XY2imageij(ptsx,ptsy)
# 		casp_rint_j=np.round(corr_array_sample_pts_j).astype('int')
# 		casp_rint_i=np.round(corr_array_sample_pts_i).astype('int')
# 		#print corr_image.img.shape, casp_rint_i.shape, casp_rint_j.shape
# 		
# 		image_corr_vals=corr_image.img[(casp_rint_j,casp_rint_i)] #fails here
# 
# 		stack_corr_values=corr_fields_max_vals[pointsij]
# 
# 		values_index = np.where((stack_corr_values<image_corr_vals))
# 		corr_fields_max_vals[(pointsij[0][values_index],pointsij[1][values_index])]=image_corr_vals[values_index]
# 		#flip 90 degree counter clockwise
# plt.figure()
# plt.imshow(corr_fields_max_vals)
# plt.colorbar()
# plt.show()