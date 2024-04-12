import os
import glob
import fcntl
import fnmatch
import shutil
import time
import stat
from pathlib import Path

class FileLock():
    def __init__(self, fpath):
        if fpath is Path:
            self.lock = fpath
        else:
            self.lock = Path(fpath)
        self.has_lock = True

    def __enter__(self):
        print('Locking file %s'%self.lock)
        self.f = open(self.lock.as_posix(),'w')
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            #self.f.write(hostname())
            self.f.flush()
        except IOError:
            print('Locking failed %s'%self.lock)
            self.has_lock = False
        return self
    def __exit__(self, type, value, traceback):
        if self.has_lock:
            print('Unlocking file %s'%self.lock)
            self.f.close()
            try:
                self.lock.unlink()
            except:
                pass



def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname)[stat.ST_MTIME]

def set_mod_time(pathname):
  # Get the current time
  current_time = time.time()

  # Set the creation and modification datetime of the file
  os.utime(pathname, (current_time, current_time))
  
  
def clear_folder_content(path):
   filelist = glob.glob(path)
   for f in filelist:
       print('deleting file '+ str(f))   
    #    os.rmdir(f)
       shutil.rmtree(f)	
  
def delete_folder(input_folder):		   
    try:
        shutil.rmtree(input_folder)
    except OSError as err:
        print("OS error: {0}".format(err))
		
   

def wait_for_file(path_file,timeout=30): #30sec wait
    start_time =time.time()
    while not os.path.exists(path_file):
        print('waiting for mesh export')
        time.sleep(3)       		
        end_time=time.time()		    
        if( end_time - start_time > timeout):
           return False

    return True	
		

def file_exists(file): 
   if not os.path.isfile(file):
      print('File ' + file + 'doesnt exist, exiting')
      return False

   return True	
   
def makeNewDir(path_dir):

    if not os.path.exists(path_dir):
       try:
           os.makedirs(path_dir,exist_ok=True)
           print("Directory " , path_dir,  " Created ")		   
       except:
             print("Directory failed  " , path_dir,  " Created ")			   
             pass	   

    else:    
        print("Directory " , path_dir,  " already exists")		
    

def get_sorted_basenames(pathFolder, exts):

   file_list = []
   
   for ext in exts:
    file_list += [os.path.basename(x) for x in glob.glob(os.path.join(pathFolder,ext))]
	

   return sorted(file_list)	
   
def read_file_as_lines(pathFile):
   if os.path.exists(pathFile):	
      with open(pathFile) as f:
           lines = f.readlines()
      return [x for x in lines if x.strip()]	   
          
       	    
   return []
   
def write_file_as_lines(pathFile,lines): 
  
   lock_file = '.cache/bg_matting_file_%s.lock'%(pathFile.split('/')[-1])	
   	   			 
   with FileLock(lock_file) as lock:
       if lock.has_lock:  
          f=open(pathFile, "w")
          
          for line in sorted(lines):    			  
             f.write(str(line))	 
			 
	   
		 
def check_if_all_obj_done(OUTPUT_DIR,frame_start,frame_end, ext = '.obj'):

   all_done = True
   
   if frame_end <= 0: return False  
   
   for iframe in range(frame_start,frame_end):
      FRAME_DIR = os.path.join(OUTPUT_DIR,"%06d"%iframe)
      if os.path.exists(FRAME_DIR):	   
         if not os.path.exists(os.path.join(FRAME_DIR, 'model'+ext)):
            all_done = False			 
            break
		 
         else: #clean up
              if(Path(str(os.path.join(FRAME_DIR, 'model'+ext))).stat().st_size < 3*1000): 
                os.remove(os.path.join(FRAME_DIR, 'model'+ext))
                all_done = False
                break				
              if os.path.exists(os.path.join(FRAME_DIR, 'output')):			 
                 try:
                     shutil.rmtree(os.path.join(FRAME_DIR, 'output'))
                 except OSError as err:
                     print("OS error: {0}".format(err))
              if os.path.exists(os.path.join(FRAME_DIR, 'transform_000000')):							 
                 try:
                     shutil.rmtree(os.path.join(FRAME_DIR, 'transform_000000'))
                 except OSError as err:
                     print("OS error: {0}".format(err))
   				 
              if os.path.exists(os.path.join(FRAME_DIR, 'temp')):						  
                 try:
                     shutil.rmtree(os.path.join(FRAME_DIR,'temp'))
                 except OSError as err:
                     print("OS error: {0}".format(err))					 
      else:
           all_done = False			 
           break	
		   
   return all_done	

def all_done_videos(dir_path,num_of_cams):	
    # list to store files
    number = 0
    # Iterate directory
    for file in os.listdir(dir_path):
       #print(file)	
        # check only text files
       if fnmatch.fnmatch(file,  'stream[0-9]*.mp4') and not fnmatch.fnmatch(file,  'stream[0-9]*_.mp4'):
          number += 1
          command_test = "ffprobe -loglevel error -show_entries stream=codec_type -of default=nw=1 " + os.path.join(dir_path,file) 
          test_result = os.popen(command_test).read()
          if "codec_type=video" not in test_result:
             number -= 1
	  
		 
    #print('number of finished videos ' + str(number))			
    return number == num_of_cams














	