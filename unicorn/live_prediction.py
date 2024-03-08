from model import read_and_process_file, get_highest_occurrence
import pickle
import UnicornPy
import struct
import time

def connect_and_acquire_data(serial_number):
    try:
        # Initialize a new instance of the Unicorn class to connect to the device.
        device = UnicornPy.Unicorn(serial_number)
        
        
        # Define the number of scans to read in each iteration to match the desired interval.
        number_of_scans = 250 # Adjust based on the actual sampling rate and desired interval.
        number_of_channels = device.GetNumberOfAcquiredChannels()
        scan_size = number_of_channels * 4  # Each channel's data is a 32-bit float.
        buffer_size = number_of_scans * scan_size
        
        # Prepare the buffer to store the data.
        data_buffer = bytearray(buffer_size)
        
        # Read in model and pipeline
        folder = ""
        svm_model_path = folder +'svm.pkl'
        with open(svm_model_path, 'rb') as file:
            loaded_svm_model = pickle.load(file)

        pipeline_path = folder +'pipeline.pkl'
        with open(pipeline_path, 'rb') as file:
            preprocess_pipeline = pickle.load(file)
        

        while True:
            device.StartAcquisition(True)

            time.sleep(1)
            # try:
            device.GetData(number_of_scans, data_buffer, buffer_size)
        
            # Convert the bytearray data to float values.
            floats = struct.unpack('f' * number_of_channels * number_of_scans, data_buffer)
            
            data = []
            data.append(["EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8", "Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z", "Battery Level", "Counter", "Validation Indicator"])
            for i in range(number_of_scans):
                row = list(floats[i*number_of_channels:(i+1)*number_of_channels])
                data.append([str(x) for x in row])
            

            print(data[1][0])
            print(f"Battery: {data[1][14]}")
            data = read_and_process_file(data)

            X_transformed = preprocess_pipeline.transform(data)
            y_pred = loaded_svm_model.predict(X_transformed)

            print(f"Predicted: {get_highest_occurrence(y_pred)}")
            counter = {"dummy": 0, "up": 0, "down": 0, "rotate": 0, "forward": 0}
            for e in y_pred:
                counter[e] += 1
            print(counter)
            print()


            time.sleep(1)

            # except Exception as e:
            #     print(f"An error occurred: {e}")

            device.StopAcquisition()
        
    except UnicornPy.DeviceException as e:
        print(f"An error occurred: {e}")

connect_and_acquire_data('UN-2023.06.06')
