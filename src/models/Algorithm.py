class Algorithm:
    def __init__(self, func):
        self.func, self.args, self.kwargs = func
        self.alg = []
        self.log = logging.getLogger(__)

    def __call__(self, *args, **kwargs):
        return self.func(*self.args, **kwargs)

    def __str__(self):
        return str(self.alg)




