"""
Resize a directory of images
"""
import sys
import os
from PIL import Image
import glob

if __name__ == "__main__":
    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        height = int(sys.argv[3])
        width  = int(sys.argv[4])

        assert os.path.isdir(input_dir), "%s not found" % input_dir

        if not (os.path.isdir(output_dir)):
            print("[!] creating %s" % output_dir)
            os.makedirs(output_dir)
        
        files = glob.glob(input_dir.strip("/") + "/*")
        for f in files:
            im = Image.open(f) 
            new_f  = output_dir + "/" + f.split("/")[-1]
            
            im2 = im.resize((width,height),Image.ANTIALIAS)

            im2.save(new_f)

    except IndexError:
        sys.exit("[Usage] %s <input_directory> <output_directory> <height> <width>" % sys.argv[0])

    except TypeError:
        sys.exit("[!!] height or width not integer valued")
