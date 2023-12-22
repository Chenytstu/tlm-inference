from threading import Thread

class TLM_Thread(Thread):
    def __init__(self, func, *args):
        super().__init__()
        self.__result = None;
        self.func = func
        self.args = args
    
    def refresh(self, func=None, args=None):
        if func:
            self.func = func
        if args:
            self.args = args

    def get_result(self):
        return self.__result
    
    def run(self):
        self.__result = self.func(*self.args)
        
class TLM_ThreadFactory():
    def __init__(self, threads):
        self.threads = threads
        
    def append(self, thread: TLM_Thread):
        self.threads.append(thread)
        
    def excute(self):
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()
            
    def get_thread_result(self, index: int):
        return self.threads[index].get_result()
            
    def get(self, index: int):
        return self.threads[index]
    
    def length(self):
        return len(self.threads)
    
    def empty(self):
        self.threads = []
    