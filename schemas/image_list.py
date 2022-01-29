class ImageList:
    def __init__(self, images = []):
        self.images = images
        print("images list")
        
    @property
    def data(self):
        return self.images
