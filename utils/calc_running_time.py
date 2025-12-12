import time

def calc_end_time(start_time, print_s):
        end_time = time.time()
        diff = end_time - start_time
        print(f"{print_s} duration: {diff}(s)")

        return end_time