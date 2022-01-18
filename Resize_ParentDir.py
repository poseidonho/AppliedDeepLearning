import os 
import glob 
import cv2 
  
if __name__ == "__main__": 
     import argparse 
  
     parser = argparse.ArgumentParser( 
         description="Resize raw images to uniformed target size." 
     ) 
     parser.add_argument( 
         "--raw-dir", 
         help="Directory path to raw images.", 
         default="D:/My Pciture 2020/ConputerVision/Applied Deep Learning/raw", 
         type=str, 
     ) 
     parser.add_argument( 
         "--save-dir", 
         help="Directory path to save resized images.", 
         default="D:/My Pciture 2020/ConputerVision/Applied Deep Learning/resize", 
         type=str, 
     ) 
     parser.add_argument( 
         "--ext", help="Raw image files extension to resize.", default="jpg", type=str 
     ) 
     parser.add_argument( 
         "--target-size", 
         help="Target size to resize as a tuple of 2 integers.", 
         default="(416, 416)", 
         type=str, 
     ) 
     args = parser.parse_args() 
  
     raw_dir = args.raw_dir 
     save_dir = args.save_dir 
     ext = args.ext 
     target_size = eval(args.target_size) 
     msg = "--target-size must be a tuple of 2 integers" 
     assert isinstance(target_size, tuple) and len(target_size) == 2, msg 

     for subfolder1 in os.listdir(raw_dir):
      os.makedirs(save_dir+"/"+subfolder1, exist_ok=True)
      print(subfolder1)
      for subfolder2 in os.listdir(raw_dir+"/"+subfolder1):
        print(subfolder2)
        read_dir = raw_dir+"/"+subfolder1+"/"+subfolder2
        fnames = glob.glob(os.path.join(read_dir, "*.{}".format(ext)))
        new_dir =  save_dir+"/"+subfolder1+"/"+subfolder2
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
        
        print( 
          "\nDone resizing {} files.\nSaved to directory: `{}`".format( 
          len(fnames), new_dir 
          ) 
        ) 
