"""BCI Eye Classifier"""

# imports 
import argparse
import numpy as np  
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import os
import bci_helper as BCI  
import socket


# INITIALIZE PARAMETERS
BUFFER_LENGTH   = 5  
EPOCH_LENGTH    = 1
OVERLAP_LENGTH  = 0.8
SHIFT_LENGTH    = EPOCH_LENGTH - OVERLAP_LENGTH
TRAINING_LENGTH = 2
INDEX_CHANNEL   = [0, 1 , 2, 3] # use all four electrodes
N_CHANNELS      = 4


if __name__ == "__main__":

    """ CONNECT TO MUSE STREAM"""
    print('Connecting...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    # set up Inlet
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Pull relevant information
    info       = inlet.info()
    desc       = info.desc()
    freq       = int(info.nominal_srate())
    

    """ RECORD DATA  """
    # Record each feature
    print('Keep Eyes Open')
    eeg_data0 = BCI.record_eeg(TRAINING_LENGTH, freq, INDEX_CHANNEL)
    
    print("Move Eyes Side to Side")
    eeg_data1 = BCI.record_eeg(TRAINING_LENGTH, freq, INDEX_CHANNEL)
   
    print("Move Eyes Up and Down")
    eeg_data2 = BCI.record_eeg(TRAINING_LENGTH, freq, INDEX_CHANNEL)

    print("Blink Rapidly ")
    eeg_data3 = BCI.record_eeg(TRAINING_LENGTH, freq, INDEX_CHANNEL)

    
    # Divide data into epochs
    eeg_epochs0 = BCI.epoch_array(eeg_data0, EPOCH_LENGTH, OVERLAP_LENGTH * freq, freq)
    eeg_epochs1 = BCI.epoch_array(eeg_data0, EPOCH_LENGTH, OVERLAP_LENGTH * freq, freq)
    eeg_epochs2 = BCI.epoch_array(eeg_data0, EPOCH_LENGTH, OVERLAP_LENGTH * freq, freq)
    eeg_epochs3 = BCI.epoch_array(eeg_data0, EPOCH_LENGTH, OVERLAP_LENGTH * freq, freq)

   
   	# Computer corresponding features
    feat_matrix0 = BCI.compute_feature_matrix(eeg_epochs0, freq)
    feat_matrix1 = BCI.compute_feature_matrix(eeg_epochs1, freq)
    feat_matrix2 = BCI.compute_feature_matrix(eeg_epochs2, freq)
    feat_matrix3 = BCI.compute_feature_matrix(eeg_epochs3, freq)
    

    # Train Classifier
    [classifier, mu_ft, std_ft, score] = BCI.train_classifier(
                                            feat_matrix0, 
                                            feat_matrix1, 
                                            feat_matrix2, 
                                            feat_matrix3, 
                                            'RandomForestClassifier')

    print(str(score * 100) + '% correctly predicted')
  

    # Initialize the buffers for real time data procession
    eeg_buffer      = np.zeros((int(freq * BUFFER_LENGTH), N_CHANNELS))
    filter_state    = None
    decision_buffer = np.zeros((30, 1))

    # PLotter to see decisions
    plotter_decision = BCI.DataPlotter(30, ['Decision'])

    #initialize socket instance
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    # create a port for the socket
    s.bind((socket.gethostname(), 1456)) 
    #listen for an avaliable client for 5 seconds
    s.listen(5)

    try:
        while True:

  
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * freq))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer
            eeg_buffer, filter_state = BCI.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

            """ 3.2 COMPUTE FEATURES AND CLASSIFY """
            # Get newest samples from the buffer
            data_epoch = BCI.get_last_data(eeg_buffer,
                                            EPOCH_LENGTH * freq)

            # Compute features
            feat_vector = BCI.compute_band_powers(data_epoch, freq)
            y_hat = BCI.test_classifier(classifier,
                                        feat_vector.reshape(1, -1), 
                                        mu_ft,
                                        std_ft)
            print(y_hat)

            decision_buffer, _ = BCI.update_buffer(decision_buffer,
                                                    np.reshape(y_hat, (-1, 1)))

            print(decision_buffer)
            #print(type(decision_buffer))

            clientsocket, address = s.accept() # accepts client and gets IP of client
            print("Connection has been established.")
            pred = np.mean(decision_buffer)
            #for i in range: (0, 16):
            x = str(pred) #str of first value of vector (0, 1, 2, or 3)
            print(x)
            clientsocket.send(bytes(x, 'utf-8')) #sends the data

           
    except KeyboardInterrupt:

        print('Closed!')
