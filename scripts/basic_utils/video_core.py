"""
video_core.py is a subsection of basic_utils library that is meant 
for video processing. 
"""

import sys
import os
import cv2
import basic_utils.basics as base

FPS = 30

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

def partition_frame_dir(frame_dir, output_dir, num_seconds=2, front_trim=60, end_trim=60, 
        capacity_tolerance=1.0):
    """
    Given a directory of frames, this method partitions them into several subdirectories,
    such that each directory contains num_seconds*FPS frames.

    args:
        frame_dir : directory containing frames
        output_dir : directory to place partitions
        num_seconds (optional) : the number of seconds that make up each partition
        front_trim (optional) : the number of frames to ignore from beginning of directory
        end_trim (optional) : the number of frames to ignore from end of directory
        capacity_tolerance (optional) : the percentage of an incomplete paritition 
                that will still be considered acceptable
    """
    
    if not os.path.exists(frame_dir):
        raise FileNotFoundError("provided directory |%s| not found" % frame_dir)
    if not os.path.isdir(frame_dir):
        raise IOError("provided path |%s| is not a directory" % frame_dir)
    if front_trim < 0 or end_trim < 0 or num_seconds < 0:
        raise ValueError("num_seconds, front_trim and end_trim must be positive")
    
    listed_directory = os.listdir(frame_dir)
    num_frames = len(listed_directory)

    #rough estimation to determine frame rate
    is_60_fps = num_frames > 1600
    
    print('[partition_frame_dir]: {} -> {}'.format(frame_dir, output_dir))
    print("[partition_frame_dir]: Found {} frames".format(num_frames), ("(60fps)" if is_60_fps else "(30fps)"))
    iteration = 0
    num_partitions = 0
    current_partition = []
    total = (num_frames - front_trim - end_trim) // (num_seconds * FPS) 
    while iteration < num_frames - end_trim:
        if iteration >= front_trim:
            if not is_60_fps:
                current_partition.append(listed_directory[iteration])
            elif iteration % 2 == 0:
                current_partition.append(listed_directory[iteration])
            
            if len(current_partition) >= num_seconds*FPS:
                next_output_dir = os.path.join(output_dir, str(num_partitions))
                move_frames(frame_dir, current_partition, next_output_dir)           
                num_partitions+=1
                current_partition = []
                progressBar(num_partitions, total)
        iteration+=1
    #Handle any leftover frame lists that did not reach full capacity due to trimming.
    #If the frame list is acceptably full, then include it with the other partitioned directories
    if current_partition:
        if len(current_partition) / (num_seconds*FPS) >= capacity_tolerance:
            next_output_dir = os.path.join(output_dir, str(num_partitions))
            move_frames(frame_dir, current_partition, next_output_dir)

def move_frames(source_dir, partitioned_frames, output_dir):
    """
    Helper method to move a selection of frames from the source directory
    into a new output partition directory.

    args:
        source_dir : the origin directory
        partitioned_frames : a list of frames that are being moved out of source directory
        output_dir : the directory where selected frames are being placed
    """
    base.check_exists_create_if_not(output_dir)
    current_index = 0
    for frame in partitioned_frames:
        frame_path = os.path.join(source_dir, frame)
        source_path = os.path.join(output_dir, "frame" + str(current_index) + ".png")
        os.rename(frame_path, source_path)
        current_index+=1

def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()



