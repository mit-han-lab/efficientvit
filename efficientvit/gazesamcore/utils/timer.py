import time

import torch

__all__ = ["Timer"]


class Timer:
    def __init__(self):
        self.times = {}
        self.counts = {}
        self.running = {}

    def start(self, category):
        torch.cuda.synchronize()

        if category in self.running:
            raise ValueError(f"Category {category} already running: {self.running.keys()}")

        self.running[category] = time.time()

    def stop(self, category):
        torch.cuda.synchronize()
        end_time = time.time()
        if category not in self.running:
            raise ValueError(f"Category {category} not found in running timers: {self.times.keys()}")

        runtime = end_time - self.running[category]
        self.times[category] = self.times.get(category, []) + [runtime]
        self.counts[category] = self.counts.get(category, 0) + 1

        del self.running[category]

    def print_average(self, category):
        total_time, total_count = sum(self.times[category]), self.counts[category]
        if self.counts[category] > 5:
            total_time -= sum(self.times[category][:5])
            total_count -= 5

        print(f"{category} avg: {total_time / total_count}")

    def print_averages_across_categories(self):
        for category in self.times:
            self.print_average(category)
