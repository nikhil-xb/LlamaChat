import os
import multiprocessing 
import glob
class preProcess:
    def __init__(self, path, your_name) -> None:
          self.path= path
          self.your_name= your_name
    def chatprocess(self):
        content= ''''''
        corrected= []
        with open(self.path, 'r') as file:
            content= file.read()
        file.close()
        lines= content.split('\n')
        for line in  lines:
            parts= line.split(':')
            if len(parts)>=3:
                text_= parts[1].strip()
                text_= text_.split('-')
                if len(text_)==2:
                    name= text_[1].strip()
                    message= parts[2].strip()
                    if name==self.your_name:
                        text_msg= name+": "+message
                        corrected.append(text_msg)
        return corrected
    
    def chatload(self,output_path):
        pool= multiprocessing.Pool(processes=4)
        txt_files= glob.glob(os.path.join(self.path,"**/*.txt"), recursive=True)
        outputs= pool.map(self.process, txt_files)
        master= []
        for x_ in outputs:
            master.extend(x_)
        with open(os.path.join(self.path, output_path),'w') as file:
                file.write('/n'.join(master))
        file.close()
