#  coding:utf-8
import subprocess
import os
import threading

def main(frame_dir, face_dir, n_thread):

    threads = []
    # function
    func_path = './lib/face_align_cuda.py'
    # Model
    predictor_path      = './lib/shape_predictor_5_face_landmarks.dat'
    cnn_face_detector   = './lib/mmod_human_face_detector.dat'
    for category in os.listdir(frame_dir):
        category_dir = os.path.join(frame_dir, category)
        
        for frame_file in os.listdir(category_dir):
            frame_root_folder = os.path.join(category_dir, frame_file)
            face_root_folder = frame_root_folder.replace(frame_dir, face_dir)

            if os.path.isdir(frame_root_folder):
                makefile(face_root_folder)
                threads.append(threadFun(frame2face, (func_path, predictor_path, frame_root_folder, face_root_folder, cnn_face_detector)))

    run_threads(threads, n_thread)
    print('all is over')

def makefile(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
def run_threads(threads, n_thread):
    used_thread = []
    for num, new_thread in enumerate(threads):
        new_thread.start()
        used_thread.append(new_thread)
        
        if num % n_thread == 0:
            for old_thread in used_thread:
                old_thread.join()
            used_thread = []
            
class threadFun(threading.Thread):
    def __init__(self, func, args):
        super(threadFun, self).__init__()
        self.fun = func
        self.args = args
    def run(self):
        self.fun(*self.args)

def frame2face(func_path, predictor_path, image_root_folder, save_root_folder, cnn_face_detector, gpu_id = 0):

    linux_command = 'python {:} {:} {:} {:} {:} {:}'.format(func_path, predictor_path, image_root_folder, save_root_folder, cnn_face_detector, gpu_id)
    subprocess.getstatusoutput(linux_command)
    print('thread {:}'.format(image_root_folder))
    
if __name__ == '__main__':
    
    frame_dir_train = '../frame/train'
    face_dir_train  = '../face/train'
    
    frame_dir_val = '../frame/val' 
    face_dir_val  = '../face/val'
    
    main(frame_dir_train, face_dir_train, n_thread=20)
    main(frame_dir_val, face_dir_val, n_thread=20)

