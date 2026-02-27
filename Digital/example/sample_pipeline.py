

import random
from Digital.core.signal_processor import SignalProcessor
from Digital.io.signal_generator import generate_noisy_signal 

def main():
    # magic numbers for signal generation
    # hardcoded for simplicity, but can be modified to take user input or be randomized
    # These values are used to generate the initial noisy signal and can be changed in each loop iteration
    sample_rate = 1000
    duration = 2
    frequency = 50
    #simulates arduino setup and loop functions
    # The hardware version of this program would run the setup function once to initialize the system and then continuously run the loop function to process signals in real-time. In this simulation, we call the setup function once to get the number of loops and then call the loop function that many times to simulate the continuous processing of signals.
    times_looped = setup() 
    #loops through the signal processing steps the specified number of times, simulating the continuous processing of signals in an embedded system. Each loop generates a new noisy signal, processes it, and visualizes the results, allowing you to see how the signal changes with each iteration.
    # The hardware version would continuously process signals in real-time, while this simulation allows you to specify how many times you want to loop through the process for testing and demonstration purposes.
    loop(times_looped, sample_rate, duration, frequency)
        
def setup():
    print("Welcome to the Signal Processor!")
    print("This program generates a noisy signal, processes it, and visualizes the results.")
    print("You can specify how many times you want to loop through the process.")
    print("ENTER HOW MANY TIMES YOU WANT TO LOOP THROUGH THE PROCESS: ")
    times_looped = int(input())
    while times_looped <= 0:
        print("Please enter a positive integer.")
        times_looped = int(input())
    return times_looped

def loop(times_looped, sample_rate, duration, frequency):
    print("Looping through the signal processing steps...")
    for i in range(times_looped):  
        signal = generate_noisy_signal(freq=frequency, duration=duration, sample_rate=sample_rate)
        sp = SignalProcessor(signal, sample_rate)
        sp.plot_time_domain()
        sp.plot_frequency_domain(xlim=200)
        sp.apply_bandpass(39, 61)
        sp.normalize()
        sp.plot_time_domain()
        sp.plot_frequency_domain(xlim=200)
        sample_rate = random.randint(800, 1200)
        sp.sample_rate = sample_rate
        duration = random.uniform(1, 3)
        frequency = random.randint(40, 60) 
if __name__ == "__main__":
    main() 