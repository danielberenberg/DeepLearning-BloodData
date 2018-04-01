import csv
import subprocess
import os
import sys
import basic_utils.basics as base

data_dir = 'data'
selected = 'NextStartingPoint.csv'
output_dir = 'augmented'

BASH_COMMAND = 'ffmpeg -i {} -vf setpts={}*PTS {}' 
GET_DURATION_COMMAND = "ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 {}"
SPLIT_COMMAND = "ffmpeg -i {} -vcodec copy -acodec copy -ss {} -t {} {} \
   -vcodec copy -acodec copy -ss {} -t {} {}"
HALVE_COMMAND = "ffmpeg -i {} -vcodec copy -acodec copy \
    -ss {} -t {} {}"

def main():
    base.check_exists_create_if_not(output_dir)

    with open(selected, 'r') as video_csv:
        csvreader = csv.reader(video_csv)
        next(csvreader) #skip the header
        for line in csvreader:
            subj_path = fetch_path(line[0])
            trial = line[1]
            trial_path = ""
            for video in os.listdir(subj_path):
                if trial in video:
                    if trial == "1":
                        speed = 0.5
                        ext = "FAST"
                        trial_path = video
                    else:
                        speed = 2
                        ext = "SLOW"
                        trial_path = video
                    break
            out = os.path.join(output_dir, subj_path.split('/')[1])
            base.check_exists_create_if_not(out)
            new_name = os.path.join(out, trial_path + '_' + ext + '.mov')
            change_speed(os.path.join(subj_path, video), new_name, speed)

def fetch_path(subj):
    subject = "S" + ("0" * (4-len(subj))) + subj
    full_path = os.path.join(data_dir, subject)
    return full_path

def change_speed(video_path, new_name, factor):
    command = BASH_COMMAND.format(video_path, factor, new_name)
    print("[{}]: CHANGING SPEED of video {} by a factor of {}".format(sys.argv[0], video_path, str(factor)))
    p = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)
    p.wait()
    p.kill()
    handle(new_name, factor)

def handle(video_path, factor):
    command = GET_DURATION_COMMAND.format(video_path)
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
   
    if factor == 2: #if the video was slowed down, split it into two 30 second parts
        split_name = video_path.split('.')[0]
        end_time_1 = int(float(out) / 2)
        end_time_2 = int(float(out))
        if len(str(end_time_1)) < 2:
            end_time_1 = int("0" + str(end_time_1))
        if len(str(end_time_2)) < 2:
            end_time_2 = int("0" + str(end_time_2))
        end_time_1 = "00:00:" + str(end_time_1)
        if end_time_2 > 60:
            end_time_2 = "00:01:" + '0' + str(end_time_2 - 60)
        else:
            end_time_2 = "00:00:" + str(end_time_2)
        command = SPLIT_COMMAND.format(video_path, 0, end_time_1, split_name + '_first_half.mov',
                end_time_1, end_time_2, split_name + '_second_half.mov') 
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        p.wait()
        p.kill()
    else: #if the video was sped up, cut it in half and reverse it
        halved_name = video_path.split('.')[0] + '_halved.mov'
        end_time = "00:00:" + str(int(float(out)/2))
        command = HALVE_COMMAND.format(video_path, "00:00:00", end_time, halved_name) 
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        looped_name = video_path.split('.')[0] + '_looped.mov'
        command = ['ffmpeg','-i',halved_name,'-filter_complex','[0:v]reverse,fifo[r];[0:v][0:a][r][0:a]concat=n=2:v=1:a=1 [v] [a]', '-map', '[v]', '-map', '[a]', looped_name] 
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        os.remove(halved_name)
    os.remove(video_path)

main()
