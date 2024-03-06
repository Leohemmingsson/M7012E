import UnicornPy
import struct
import time

def connect_and_acquire_data(serial_number):
    try:
        # Initialize a new instance of the Unicorn class to connect to the device.
        device = UnicornPy.Unicorn(serial_number)
        
        # Start data acquisition.
        device.StartAcquisition(False)
        
        # Define the number of scans to read in each iteration to match the desired interval.
        number_of_scans = 1250 # Adjust based on the actual sampling rate and desired interval.
        number_of_channels = device.GetNumberOfAcquiredChannels()
        scan_size = number_of_channels * 4  # Each channel's data is a 32-bit float.
        buffer_size = number_of_scans * scan_size
        
        # Prepare the buffer to store the data.
        data_buffer = bytearray(buffer_size)
        
        while True:
            print("Ready:")
            input()
            time.sleep(5)
            # Read the data into the buffer.
            device.GetData(number_of_scans, data_buffer, buffer_size)
            
            # Convert the bytearray data to float values.
            floats = struct.unpack('f' * number_of_channels * number_of_scans, data_buffer)
            
            # Print the acquired data in a human-readable form.
            # This will print 25 lines of data for each channel per acquisition.
            for i in range(number_of_scans):
                print("Scan", i+1, ":", floats[i*number_of_channels:(i+1)*number_of_channels])
        
        # Note: In this example, the acquisition does not stop automatically.
        # You should implement a condition to break out of the loop and stop the acquisition.
        
    except UnicornPy.DeviceException as e:
        print(f"An error occurred: {e}")



connect_and_acquire_data('UN-2023.06.06')
