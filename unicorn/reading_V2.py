import UnicornPy
import struct
import time
import csv
import random

def connect_and_acquire_data(serial_number):
    try:
        # Initialize a new instance of the Unicorn class to connect to the device.
        device = UnicornPy.Unicorn(serial_number)
        
        # Start data acquisition.
        device.StartAcquisition(False)
        
        # Define the number of scans to read in each iteration to match the desired interval.
        number_of_scans = 250 # Adjust based on the actual sampling rate and desired interval.
        number_of_channels = device.GetNumberOfAcquiredChannels()
        scan_size = number_of_channels * 4  # Each channel's data is a 32-bit float.
        buffer_size = number_of_scans * scan_size
        
        # Prepare the buffer to store the data.
        data_buffer = bytearray(buffer_size)
        
        
        all_choices = {}
        filepath = "unicorn/data/"

        while True:
            command = random.choice(["up", "down", "rotate", "forward"])
            if command not in all_choices:
                all_choices[command] = 0
            all_choices[command] += 1

            print(command)
            print("Press Enter to start data acquisition...")
            input()
            time.sleep(1)
            device.GetData(number_of_scans, data_buffer, buffer_size)
            
            # Convert the bytearray data to float values.
            floats = struct.unpack('f' * number_of_channels * number_of_scans, data_buffer)
            
            # Prepare data for CSV.
            csv_data = []
            column_names = ["EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8", "Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation", "Indicator"]
            for i in range(number_of_scans):
                row = list(floats[i*number_of_channels:(i+1)*number_of_channels])
                csv_data.append(row)
            
            number = all_choices[command]
            output_file_path = f"{filepath}{command}{number}.csv"
            
            # Write data to CSV.
            with open(output_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(column_names)  # Write the column names
                writer.writerows(csv_data)  # Write the data rows
            
            print(f"Data acquisition complete. Data written to {output_file_path}")
        
    except UnicornPy.DeviceException as e:
        print(f"An error occurred: {e}")

connect_and_acquire_data('UN-2023.06.06')
