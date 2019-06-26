from timeit import default_timer as timer

class print_time():
    """A class used to count time in methods
    Usage
    -----
    t = print_time()
    t.reset()
    # run some process
    t.time() # prints elapsed time
    # run another process
    t.time() # prints elapsed time since last checked
    t.total() # prints total time elapsed since last reset
    """

    def __init__(self):
        self.start = 0
        self.checkpoint = [0]
    
    def reset(self):
        """Resets timer and checkpoints when called"""

        self.start = timer()
        self.checkpoint = [self.start]
        
    def _format_time(self, sec):
        """Parses seconds in HH:MM:SS format

        Parameters
        ---------
        sec : float
            Current timestamp in seconds

        Returns
        -------
        str : timestamp string formatted in HH:MM:SS
        """

        secs = round(sec % 60)
        mins = round((sec // 60) % 60)
        hrs = round(sec // 60 // 60)
        
        stamp = '{:02d}:{:02d}:{:02d}'.format(hrs, mins, secs)
        
        return stamp
        
    def time(self):
        """Prints time in HH:MM:SS when called"""

        self.start = self.checkpoint[-1]
        self.checkpoint.append(timer())
        print(self._format_time(self.checkpoint[-1] - self.start))

    def total(self):
        """Prints total time elapsed when called
        Note: This method does not leave a checkpoint
        """

        print(self._format_time(timer()-self.checkpoint[0]))
