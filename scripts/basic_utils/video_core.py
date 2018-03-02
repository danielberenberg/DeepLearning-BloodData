"""
video_core.py is a subsection of basic_utils library that is meant 
for video processing. 
"""

import sys
import os
import cv2
import basic_utils.basics as base


def video_file_exists(filename):
    """
    Return whether or not this file exists and is a
    video file (i.e, ends in [.MOV | .mov]

    args:
        --> filename : the file to test

    returns:
        --> boolean of whether this file is a video file
        --> error message if something bad occurred
        --> the filename minus the extension and parent dirs
    """
    does_exist = os.path.exists(filename)
    is_vid = filename.endswith(".mov") or filename.endswith(".MOV")


    if does_exist and is_vid:
        parent = filename.split("/")[-2]
        no_ext = filename.split("/")[-1][:-4]
        
        # example : S0001/Trial1
        return_me = os.path.join(parent, no_ext)
        return True, None, return_me

    output_string = []
    if not is_vid:
        output_string.append("%s isn't a .MOV or .mov" % filename)

    if not does_exist:
        output_string.append("%s not found" % filename)

    error = " AND ".join(output_string)
    return False, error, None


def video_file_to_frames(filename, output_dir=None, suppress=False):
    """
    Convert a video file to individual frames

    args:
        --> filename : video file to convert
        --> output_dir (optional): the desired output directory
                               for the frames

        --> suppress : boolean to suppress messages or not
    returns:
        --> list of image filenames
    some facts:
    ----------
        1) This procedure will save the png images in an output directory
           called '$(filename_dir)/$(filename_frames)' if no directory is specified.

        2) if an output directory IS specified but not found, this procedure
           will create the output directory, with a subdir called %filename_frames

        3) Stack overflow source:
        [https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames]
    """
    # check the video's existence
    vid_valid, err, no_ext = video_file_exists(filename)
    #print("vid : {}".format(vid_valid))
    
    if vid_valid:
        if output_dir:
            output_dir = os.path.join(output_dir, "%s_frames" % no_ext)
            base.check_exists_create_if_not(output_dir, suppress=suppress)

        else:
            output_dir = "%s_frames" % no_ext
            base.check_exists_create_if_not(output_dir, suppress=suppress)

        # have output directory, now need to create the framesies
        vidcap = cv2.VideoCapture(filename)
        success, image = vidcap.read()
        
        image_names = []
        count = 0
        success = True

        # while there is a next image
        while success:
            pth = os.path.join(output_dir, "frame-%05d.png" % count)
            image_names.append(pth)
            cv2.imwrite(pth, image)
            success, image = vidcap.read()
            
            if not suppress:
                if success:
                    sys.stdout.write("\r[video_file_to_frames]-- writing [%s]" % pth)
                    sys.stdout.flush()
                
                else:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            count += 1

        return image_names
    # a problem occurred
    else:
        raise ValueError(err)

def video_dir_to_frame_dir(video_dir, output_dir, suppress=False):
    """
    create a directory that contains subdirectories
    that contain all of the frames of the movies contained
    within the video_dir

    args:
        video_dir : directory with videos
        output_dir : location to place the output frames
        suppress (optional) : display output 
    returns:
        imgs_captured : a list of image framenames from all 
                        videos in video dir
    """
    
    # path exists?
    if os.path.exists(video_dir):
        
        # path is a dir?
        if os.path.isdir(video_dir):
            contents = os.listdir(video_dir)
            movies = [os.path.join(video_dir, cont) for cont in contents if cont.lower().endswith(".mov")]

            # path contains .MOV or .mov files ??
            if len(movies) > 0:
                
                imgs_captured = []

                base.check_exists_create_if_not(output_dir, suppress=suppress)

                for mov in movies:
                    framenames = video_file_to_frames(mov, output_dir=output_dir, suppress=suppress)
                    imgs_captured.extend(framenames)

                return imgs_captured

            else:
                raise FileNotFoundError("%s contains no video files" % video_dir)

        else:
            raise ValueError("%s isn't a directory" % video_dir)

    else:
        raise FileNotFoundError("%s not found" % video_dir)
