import functions.feature_handler as fh
import functions.data_handler as dh


class Buffer:
    def __init__(self, content, MAX_LENGTH):
        self._content = list(content)
        self._length = MAX_LENGTH

    # adds data to the buffer while remaining within maximum length (like a real buffer)
    def add(self, motion):
        self._content.append(motion)
        self.update()

    # prevents the buffer from exceeding the maximum length
    def update(self):
        if len(self._content) > self._length:
            self._content = self._content[len(self._content) - self._length:]

    # returns a filtered version of its contents
    def filter(self, TIME_PERIOD, zero_phase=True):
        return dh.filter_data(self._content, TIME_PERIOD, zero_phase)

    # returns a normalised version of its contents
    def normalise(self):
        return fh.normalise(self._content)

    # returns midpoint and spread of its contents
    def get_data_attributes(self):
        return fh.get_norm_attributes(self.content)  # returns [mid, sigma]

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, data):
        self._content = list(data)
        self.update()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.update()
