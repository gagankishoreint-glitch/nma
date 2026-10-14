"""CSV is mandatory. TensorBoard is optional, wandb never required."""
import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path, fields):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def log(self, **row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
