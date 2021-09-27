from collections import defaultdict
from typing import Tuple

class Table:
    def __init__(self, width=1, height=1):
        self.width = width
        self.height = height

        # indices of rows: 0 is the top line, self.height after the last row
        self.hlines = []

        # keys: tuples of (row, col)
        self._data = defaultdict(str)
        
    def __getitem__(self, pos):
        row, col = pos
        return self._data[(row, col)]

    def __setitem__(self, pos, data):
        row, col = pos
        self._data[(row, col)] = data

    def __str__(self):
        table = ""
        for row in range(self.height):
            if row in self.hlines:
                table += "\\hline\n"
            r = [str(self._data[(row, col)]) for col in range(self.width)]
            table += " & ".join(r) + "\\\\\n"
        
        if row+1 in self.hlines:
            table += "\\hline\n"

        return table

    def __repr__(self) -> str:
        return self.__str__()

    def __repr_html__(self) -> str:
        return self.__str__()
