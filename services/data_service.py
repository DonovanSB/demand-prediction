def singleton(cls):
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper


@singleton
class DataService:
    def __init__(self) -> None:
        self.df = None

    @property
    def data_frame(self):
        return self.df

    @data_frame.setter
    def data_frame(self, df):
        self.df = df
