import os 
import glob 
import cv2 
from PIL import Image, ImageEnhance
    
if __name__ == "__main__": 
     import argparse 
  
     parser = argparse.ArgumentParser( 
         description="Resize raw images to uniformed target size." 
     ) 
     parser.add_argument( 
         "--raw-dir", 
         help="Directory path to raw images.", 
         default="./data/raw", 
         type=str, 
     ) 
     parser.add_argument( 
         "--save-dir", 
         help="Directory path to save resized images.", 
         default="./data/images", 
         type=str, 
     ) 
     parser.add_argument( 
         "--ext", help="Raw image files extension to resize.", default="jpg", type=str 
     ) 
     parser.add_argument( 
         "--target-size", 
         help="Target size to resize as a tuple of 2 integers.", 
         default="(800, 600)", 
         type=str, 
     ) 
     parser.add_argument( 
         "--factor", 
         help="Target factor to adjust image brightness.", 
         default=1.0,
         type=float, 
     ) 
     args = parser.parse_args() 
  
     raw_dir = args.raw_dir 
     save_dir = args.save_dir 
     ext = args.ext 
     target_size = eval(args.target_size)
     factor = args.factor

     msg = "--target-size must be a tuple of 2 integers" 
     assert isinstance(target_size, tuple) and len(target_size) == 2, msg 


     for subfolder2 in os.listdir(raw_dir):
        print(subfolder2)
        read_dir = raw_dir+"/"+subfolder2
        fnames = glob.glob(os.path.join(read_dir, "*.{}".format(ext)))
        new_dir =  save_dir+"/"+subfolder2
        os.makedirs(new_dir, exist_ok=True) 
        print( 
              "{} files to resize from directory `{}` to target size:{}".format( 
              len(fnames), raw_dir, target_size  		
              ) 
        )
 
        for i, fname in enumerate(fnames): 
          print(".", end="", flush=True) 
          img = cv2.imread(fname) 
          img_small = cv2.resize(img, target_size) 
          new_fname = "{}.{}".format(str(i), ext) 
          small_fname = os.path.join(new_dir, new_fname) 
          cv2.imwrite(small_fname, img_small)
          im = Image.open(small_fname)
          #image brightness enhancer
          enhancer = ImageEnhance.Brightness(im)
          im_output = enhancer.enhance(factor)
          new_fname2 = "{}_F{}.{}".format(str(i),factor, ext)
          small_fname2 = os.path.join(new_dir, new_fname2)
          im_output.save(small_fname2)
          im_output = enhancer.enhance(factor+0.5)
          new_fname3 = "{}_F{}.{}".format(str(i),factor+0.5, ext)
          small_fname3 = os.path.join(new_dir, new_fname3)
          im_output.save(small_fname3)


        
        print( 
          "\nDone resizing {} files.\nSaved to directory: `{}`".format( 
          len(fnames), new_dir 
          ) 
        ) 
