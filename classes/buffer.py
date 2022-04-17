import functions.feature_handler as fh
import functions.data_handler as dh


class Buffer:
    def __init__(self, x, y, z, MAX_LENGTH):
        self._x = x
        self._y = y
        self._z = z
        self._length = MAX_LENGTH

    # adds data to the buffer while remaining within maximum length (like a real buffer)
    def add(self, motion):
        self._x.append(motion[0])  # X
        self._y.append(motion[1])  # Y
        self._z.append(motion[2])  # Z

        # prevents the buffer from exceeding the maximum length
        if len(self._x) > self._length or len(self._y) > self._length or len(self._z) > self._length:
            self._x = self._x[len(self._x) - self._length:]
            self._y = self._y[len(self._y) - self._length:]
            self._z = self._z[len(self._z) - self._length:]

    # returns a filtered version of its contents
    def filter(self, TIME_PERIOD):
        f_x = dh.filter_data(self._x, TIME_PERIOD)
        f_y = dh.filter_data(self._y, TIME_PERIOD)
        f_z = dh.filter_data(self._z, TIME_PERIOD)
        return [f_x, f_y, f_z]

    # returns a normalised version of its contents
    def normalise(self):
        n_x = fh.normalise(self._x)
        n_y = fh.normalise(self._y)
        n_z = fh.normalise(self._z)
        return [n_x, n_y, n_z]

    @property
    def content(self):
        return [self._x, self._y, self._z]

    @content.setter
    def content(self, data):
        self._x = data[0]
        self._y = data[1]
        self._z = data[2]

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
