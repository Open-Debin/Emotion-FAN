import subprocess
import os
import threading
import pdb
import copy

def main(video_dir, frame_dir, n_thread):
    print('convert videos into frames\nvideo_dir: {:}\tframe_dir: {:}'.format(video_dir, frame_dir))
    threads = []
    for root, dirs, files in os.walk(video_dir):
        for file_name in files:
            file_type = file_name.split('.')[-1]
            if file_type in ['mp4', 'webm', 'avi']:
                video_name = os.path.join(root, file_name)
                frame_output_path = video_name.replace(video_dir, frame_dir).replace('.'+file_type,'')
                if not os.path.exists(frame_output_path):
                    os.makedirs(frame_output_path)

                t = threadFun(video2frame, (video_name, frame_output_path ))
                threads.append(t)
    run_threads(threads, n_thread)
    print('all threads is finished')    


def run_threads(threads, n_thread):
    used_thread = []
    for num, new_thread in enumerate(threads):
        print('thread index: {:}'.format(num), end=' \t')
        new_thread.start()
        used_thread.append(new_thread)
        
        if num % n_thread == 0:
            for old_thread in used_thread:
                old_thread.join()
            used_thread = []

    
def video2frame(video_input, frame_output):
    linux_commod = 'ffmpeg -i {:} -f image2 {:}/%07d.jpg'.format(video_input, frame_output)
    print('{:}'.format(video_input))
    subprocess.getstatusoutput(linux_commod)
    
class threadFun(threading.Thread):
    def __init__(self, func, args):
        super(threadFun, self).__init__()
        self.fun = func
        self.args = args
    def run(self):
        self.fun(*self.args)

if __name__ == '__main__':
    video_dir_train ='../video/train/'
    frame_dir_train = '../frame/train/'
    
    video_dir_val ='../video/val/'
    frame_dir_val = '../frame/val/'
    
    main(video_dir_train, frame_dir_train, n_thread =20)
    main(video_dir_val, frame_dir_val, n_thread = 20)