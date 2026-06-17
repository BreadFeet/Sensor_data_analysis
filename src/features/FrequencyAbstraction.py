##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 06-01-2023

import numpy as np


# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
class FourierTransformation:

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    def find_fft_transformation(self, data):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):

        # Create new columns for the frequency data.
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_max_ampl_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size-1, len(data_table)): 
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[i - (window_size-1) : i + 1]    
                )
                # We only look at the real part in this implementation.
                for j in range(len(freqs)):
                    data_table.loc[
                        i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]
                
                # Select the dominant frequency with biggest amplitude. We only consider the positive frequencies for now.
                data_table.loc[i, col + "_max_ampl_freq"] = freqs[
                    np.argmax(real_ampl)
                ]

                # Weighted frequency calculation
                data_table.loc[i, col + "_freq_weighted"] = np.sum(freqs * real_ampl) / np.sum(real_ampl)

                # Power Spectral Density calculation
                PSD = np.divide(np.square(real_ampl), len(real_ampl))   # Energy = Amplitude^2
                PSD_pdf = np.divide(PSD, np.sum(PSD))   # Normalisation
                # Power Spectral Entropy calculation
                data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

        return data_table