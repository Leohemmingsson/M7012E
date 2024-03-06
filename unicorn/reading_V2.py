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
        
        # Define the number of scans to read in each iteration to match the desired interval.
        number_of_scans = 250 # Adjust based on the actual sampling rate and desired interval.
        number_of_channels = device.GetNumberOfAcquiredChannels()
        scan_size = number_of_channels * 4  # Each channel's data is a 32-bit float.
        buffer_size = number_of_scans * scan_size
        
        # Prepare the buffer to store the data.
        data_buffer = bytearray(buffer_size)
        
        
        all_choices = {}
        filepath = "unicorn/data/"
        retry = False
        counter = 1

        while True:
            if not retry:
                command = random.choice(["dummy", "up", "down", "rotate", "forward"])
                if command not in all_choices:
                    all_choices[command] = 0
                all_choices[command] += 1

                print(command)
                print("Press Enter to start data acquisition...")
                input()

            device.StartAcquisition(True)

            time.sleep(1)
            try:
                device.GetData(number_of_scans, data_buffer, buffer_size)
            
                # Convert the bytearray data to float values.
                floats = struct.unpack('f' * number_of_channels * number_of_scans, data_buffer)
                
                # Prepare data for CSV.
                csv_data = []
                column_names = ["EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8", "Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation Indicator"]
                for i in range(number_of_scans):
                    row = list(floats[i*number_of_channels:(i+1)*number_of_channels])
                    csv_data.append([str(x) for x in row])
                
                number = all_choices[command]
                output_file_path = f"{filepath}{command}{number}.csv"

                print(f"Battery: {csv_data[0][14]}")
                print(f"Iteration: {counter}")

                device.StopAcquisition()
                

                # Write data to CSV.
                with open(output_file_path, 'w') as file:
                    file.write("")

                with open(output_file_path, 'a') as file:
                    file.write(','.join(column_names))
                    file.write('\n')
                    for row in csv_data:
                        file.write(','.join(row))
                        file.write('\n')
                
                print(f"Data acquisition complete. Data written to {output_file_path}")
                retry = False
                counter += 1
            except Exception as e:
                print("error")
                retry = True

        
    except UnicornPy.DeviceException as e:
        print(f"An error occurred: {e}")

connect_and_acquire_data('UN-2023.06.06')
