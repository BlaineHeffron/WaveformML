import h5py


class H5FileHandler:
    def __init__(self, path, *args, **kwargs):
        self.path = path
        try:
            self.h5f = h5py.File(path, *args, **kwargs)
        except Exception as e:
            print("Opening {} failed:\n".format(self.path))
            print(e)

    def __iter__(self, *args, **kwargs):
        return self.path.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.path.__getitem__(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        return self.path.__getattr__(*args, **kwargs)
