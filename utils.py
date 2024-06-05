import csv

class SimpleCSVHandler:
    def __init__(self, filename, delimiter=','):
        self.filename = filename
        self.delimiter = delimiter
        self.file = open(self.filename, mode='w', newline='')
        self.writer = None

    def write_header(self, header):
        if self.writer is None:
            self.writer = csv.writer(self.file, delimiter=self.delimiter)
        self.writer.writerow(header)

    def write_row(self, row):
        if self.writer is None:
            self.writer = csv.writer(self.file, delimiter=self.delimiter)
        self.writer.writerow(row)

    def close(self):
        self.file.close()