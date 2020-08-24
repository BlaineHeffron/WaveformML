import h5py


class H5FileHandler:
    def __init__(self, path, *args, **kwargs):
        self.path = path
        try:
            self.h5f = h5py.File(self.path, *args, **kwargs)
        except Exception as e:
            print("Opening {} failed:\n".format(self.path))
            print(e)

    def __iter__(self, item):
        return self.h5f.__iter__(item)

    def __getitem__(self, name):
        return self.h5f.__getitem__(name)

    def __getattr__(self, attr):
        return self.h5f[attr]

    def __enter__(self, *args, **kwargs):
        return self.h5f

    def __exit__(self, *args, **kwargs):
        self.h5f.__exit__(*args, **kwargs)
