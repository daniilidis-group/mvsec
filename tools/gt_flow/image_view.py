class ImageView: 
    def __init__(self): 
        self.img = None 
        self.shape = None 
        self.cache = False 

    def set_cache(self,cache): 
        self.cache = cache

    def release(self): 
        """ Release the image data from memory """
        self.img = None
    
    def acquire(self): 
        """ Load the image data to memory """
        raise NotImplementedError()
    
    def __call__(self):
        """ Return the image array """ 
        self.acquire()
        self.shape = self.img.shape
        img = self.img
        if not self.cache: 
            self.release()
        return img
    
    def set_shape(self, shape):
        self.shape = shape
# Author Cody Phillips
