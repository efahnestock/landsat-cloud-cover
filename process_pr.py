import os, sys, glob
import argparse
		
# set up command line arguments
parser = argparse.ArgumentParser( 
    description="""Generates cloud masks for all files in a given path row directory.  
    """,
#     epilog='>>  <<',
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-band8_dir', 
                    action='store', 
                    type=str, 
                    default='.',
                    help='top band 8 directory - directory to find p???_r??? directory containing specified band 8 image in [%(default)s]')
parser.add_argument('-velocity_data_dir', 
					action='store', 
					type=str, 
					default='.',
	    			help='top directory to find p???_r??? directories containing velocity .nc files in [%(default)s]')
parser.add_argument('-path', 
                    action='store', 
                    type=str, 
                    default='0',
                    help='path input to process')
parser.add_argument('-row', 
                    action='store', 
                    type=str, 
                    default='0',
                    help='row input to process')
parser.add_argument('-output_dir', 
					action='store', 
					type=str,
					default='.', 
					help='directory to put max correlation output geotiff in [%(default)s]')
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
parser.add_argument('-output_name_tail', 
                    action='store', 
                    type=str,
                    default='_max_corr_array', 
                    help='end of name for max correlation output geotiff [%(default)s]')
args = parser.parse_args()



# os.chdir("/Users/aknh9189/Downloads/Landsat_front_tracing/L8_B8/p233_r016/")
files = glob.glob(args.band8_dir + "/p%s_r%s/*.TIF"%(args.path,args.row))
os.chdir('/Users/aknh9189/Dropbox/ethan')
for input_image in files:
    os.system('python composite_correlation_mask_L8_v1.py -band8_dir {0} -velocity_data_dir {1} -output_dir {2} {3}'.format(args.band8_dir, args.velocity_data_dir, args.output_dir+'cloud_mask_output/', os.path.basename(input_image)))
    
    