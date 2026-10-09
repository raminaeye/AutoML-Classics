'''
Author: Ramin Anushiravani
Date: April 11th/23
Data Uititly
'''

from IPython.display import clear_output
import warnings
import tensorflow
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import umap
import pickle
import os
import librosa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class DataUtil:
    '''
    This class contains method for loading, cleaning, and doing EDA on data
    '''

    def __init__(self, data=None, num_sensors=1024):
        self.sensor_i = data
        self.num_sensors = num_sensors

    def set_data(self, data):
        '''
        Sets the sensor data this object operates on.

        Args:
            data: a 1-D numpy array of sensor readings.
        '''
        self.sensor_i = data

    def set_num_sensors(self, num_sensors):
        '''
        Sets how many sensors (columns) are in the time series data.

        Args:
            num_sensors: number of sensors.
        '''
        self.num_sensors = num_sensors

    def get_mean(self):
        return np.mean(self.sensor_i)

    def get_max(self):
        return np.max(self.sensor_i)

    def get_min(self):
        return np.min(self.sensor_i)

    def get_std(self):
        return np.std(self.sensor_i)

    def get_nan(self):
        return np.sum(np.isnan(self.sensor_i))

    def get_zero_count(self):
        return np.sum(self.sensor_i == 0)

    def get_zero_nzero(self):
        return self.get_zero_count() / len(self.sensor_i)

    def capture_sensor_stat(self, times_series_data):
        '''
        Extracts per-sensor summary statistics from a time series.

        Args:
            times_series_data: matrix of shape (num_samples, num_sensors).

        Returns:
            A pandas DataFrame with one row per sensor and columns
            sensor, mean, std, ztnz (zero to non-zero ratio), max_val,
            min_val, nan_count.
        '''
        sensor_stats = []
        for i in range(0, self.num_sensors):
            self.set_data(times_series_data[:, i])
            mean_sensor_i = self.get_mean()
            max_val_i = self.get_max()
            min_val_i = self.get_min()
            nan_count_i = self.get_nan()
            std_sensor_i = self.get_std()
            num_zeros_sensor_i = self.get_zero_nzero()
            zero_to_nonzero_senor_i = self.get_zero_nzero()
            stats_senor_i = {
                "sensor": i,
                "mean": mean_sensor_i,
                "std": std_sensor_i,
                "ztnz": zero_to_nonzero_senor_i,
                "max_val": max_val_i,
                "min_val": min_val_i,
                "nan_count": nan_count_i}
            sensor_stats.append(stats_senor_i)

        df_sensor_stat = pd.DataFrame(sensor_stats)

        return df_sensor_stat

    def plot_eda(self, sampled_data):
        '''
        Plots an exploratory overview of sampled sensor data: a raw sample,
        a normalized sample, a 64-component PCA sample, an engineered-feature
        sample, aggregate statistics across sensors, and detected peaks.

        Args:
            sampled_data: matrix of shape (num_samples, num_sensors).
        '''
        avg_of_all_samples = np.mean(sampled_data, axis=0)
        std_of_all_samples = np.std(sampled_data, axis=0)
        max_of_all_samples = np.max(sampled_data, axis=0)

        sample = sampled_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample)
        plt.title("Sample sensor data")

        norm_scaler = self.normalizer(sampled_data)
        sampled_norm_data = norm_scaler.transform(sampled_data)

        sample_norm = sampled_norm_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample_norm)
        plt.title("Sample normalized sensor data")

        pca_model = self.project_pca(sampled_norm_data, dim=64)
        pca_data = pca_model.transform(sampled_norm_data)
        sample_pca = pca_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample_pca)
        plt.title("Sample 64 PCA-ed sensor data")

        eng_data = self.feature_engr(sampled_norm_data)
        sample_eng = eng_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample_eng)
        plt.title("Sample ENG-ed sensor data")

        plt.figure(figsize=(15, 5), dpi=100)
        plt.imshow(sampled_data, origin='lower', cmap='gray_r')
        plt.plot(avg_of_all_samples + 5, 'r')
        plt.plot(std_of_all_samples + 15, 'g')
        plt.plot(max_of_all_samples + 25, 'k')
        plt.legend(['mean', 'std', 'max'])
        plt.xlabel("sensors")
        plt.title("Sampled Data Stats")

        plt.figure(figsize=(15, 5), dpi=100)
        peaks_mu, _ = find_peaks(avg_of_all_samples, height=5)
        plt.plot(avg_of_all_samples)
        plt.plot(peaks_mu, avg_of_all_samples[peaks_mu], "x")
        plt.plot(std_of_all_samples)
        print("peak mu >5 {}".format(peaks_mu))
        plt.title("Sampled Data Peaks")
        plt.xlabel("sensors")
        plt.show()

    def stat_window(self, data, jump=128):
        '''
        Slides a window over the sensors, averaging them to surface patterns
        of interest. Uses 75% overlap between consecutive windows.

        Args:
            data: 1-D array of sensor readings.
            jump: window width in sensors.

        Returns:
            Two lists: window means and window standard deviations.
        '''
        b = [0, jump]
        mean_chunk_i = []
        std_chunk_i = []
        while b[1] <= len(data):
            chunck_i = data[b[0]:b[1]]
            self.set_data(chunck_i)
            mean_chunk_i.append(self.get_mean())
            std_chunk_i.append(self.get_std())
            b[0] += int(jump / 4)
            b[1] += int(jump / 4)

        return mean_chunk_i, std_chunk_i

    def make_sparse(self, data):
        '''
        Converts a dense matrix to a scipy CSR sparse matrix.

        Args:
            data: dense matrix.

        Returns:
            scipy.sparse.csr_matrix of the data.
        '''
        return scipy.sparse.csr_matrix(data)

    def feature_engr(self, ys):
        '''
        Extracts audio-style features (zero-crossing rate, RMS, spectral
        flatness, each at two frame sizes) from each row of time series data
        and applies a tanh nonlinearity.

        Args:
            ys: matrix of shape (num_samples, num_sensors) or a single 1-D row.

        Returns:
            Feature matrix of shape (num_samples, num_features).
        '''
        feats = []
        if len(np.shape(ys)) < 1:
            ys = [ys]
        for idx in range(np.shape(ys)[0]):
            y = ys[idx, :]
            zx = librosa.feature.zero_crossing_rate(
                y, frame_length=128, hop_length=64)[0]
            rms = librosa.feature.rms(y=y, frame_length=128, hop_length=64)[0]
            spec_flat = librosa.feature.spectral_flatness(
                y=y, n_fft=128, hop_length=64)[0]
            zxF = librosa.feature.zero_crossing_rate(y)[0]
            rmsF = librosa.feature.rms(y=y)[0]
            spec_flatF = librosa.feature.spectral_flatness(y=y)[0]
            self.set_data(y)
            feature = np.concatenate(
                (zx, rms, spec_flat, zxF, rmsF, spec_flatF))
            feats.append(np.tanh(feature))
        return np.array(feats)

    def project_tsne(self, data, dim=3):
        '''
        Projects data to a lower dimension using t-SNE.

        Args:
            data: matrix of shape (num_samples, num_features).
            dim: target dimensionality.

        Returns:
            Projected matrix of shape (num_samples, dim).
        '''
        data_transformation = TSNE(
            n_components=dim,
            init='random',
            perplexity=30,
            n_iter=400,
            learning_rate=5).fit_transform(data)
        return data_transformation

    def project_umap(self, data, dim=3):
        '''
        Projects data to a lower dimension using UMAP.

        Args:
            data: matrix of shape (num_samples, num_features).
            dim: target dimensionality.

        Returns:
            Projected matrix of shape (num_samples, dim).
        '''
        mapper = umap.UMAP(n_components=dim, init='random').fit(data)
        data_transformation = mapper.transform(data)
        return data_transformation

    def project_pca(self, data, dim=3):
        '''
        Fits a PCA model on the data.

        Args:
            data: matrix of shape (num_samples, num_features).
            dim: number of principal components.

        Returns:
            A fitted sklearn PCA model. Use its transform() method on new data.
        '''
        pca = PCA(n_components=dim)
        return pca.fit(data)
    
    def fit_nmf(self, data, dim=128):
        '''
        Fits a Non-Negative Matrix Factorization model on the data.

        Args:
            data: matrix of shape (num_samples, num_features). Must be non-negative.
            dim: number of NMF components.

        Returns:
            A fitted sklearn NMF model. Use its transform() method on new data.
        '''
        model = NMF(n_components=dim, init='random', random_state=0)
        return model.fit(data)

    def project_nmf(self, model, data):
        '''
        Projects data using a fitted NMF model.

        Args:
            model: a fitted sklearn NMF model (see fit_nmf).
            data: matrix to project.

        Returns:
            The projected data of shape (num_samples, dim).
        '''
        return model.transform(data)

    def scatter_plot(self, data_transformation, labels):
        '''
        Args:
            data_transformation: a 3 dim matrix
            labels : associated labels to each sample in data_transformation
        Help for scatter plot of lower dim data
        '''
        if data_transformation.shape[1] == 3:
            x = data_transformation[:, 0]
            y = data_transformation[:, 1]
            z = data_transformation[:, 2]

            x0 = np.min(x) - np.median(x)
            x1 = np.max(x) - np.median(x)
            y0 = np.min(y) - np.median(y)
            y1 = np.max(y) - np.median(y)
            z0 = np.min(z) - np.median(z)
            z1 = np.max(z) - np.median(z)
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                               marker=dict(
                                                   size=2,
                                                   color=labels,
                                                   colorscale='Viridis',
                                               ),
                                               text=labels,
                                               name='test')])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        nticks=4, range=[
                            x0, x1],), yaxis=dict(
                        nticks=4, range=[
                            y0, y1],), zaxis=dict(
                        nticks=4, range=[
                            z0, z1],),), width=700, margin=dict(
                    r=20, l=10, b=10, t=10))
            fig.show()
        elif data_transformation.shape[1] == 2:
            x = data_transformation[:, 0]
            y = data_transformation[:, 1]
            fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers',
                                             marker=dict(
                                                 size=2,
                                                 color=labels,
                                                 colorscale='Viridis',
                                             ),
                                             text=labels,
                                             name='test')])
            fig.show()

    def extract_stft(self, data, n_fft=32):
        '''
        Computes the short-time Fourier transform of a time series.

        Args:
            data: 1-D time series.
            n_fft: FFT window size.

        Returns:
            Tuple (magnitude, phase) of the STFT.
        '''
        S = librosa.stft(data, n_fft=n_fft, hop_length=n_fft // 2)
        magnitude, phase = librosa.magphase(S)
        return magnitude, phase

    def extract_psd(self, magnitude, phase):
        '''
        Extracts and plots the power spectral density from an STFT.

        Args:
            magnitude: STFT magnitude matrix from extract_stft.
            phase: STFT phase matrix from extract_stft.

        Returns:
            The average magnitude across frames (PSD estimate).
        '''
        phase_angle = np.angle(phase)
        librosa.display.specshow(magnitude, cmap='gray_r')
        avg_magnitude = np.mean(magnitude, axis=0)
        return avg_magnitude

    def load_csv(self, filename):
        '''
        Loads a CSV file and forward-fills missing values from previous rows.

        Args:
            filename: path to the CSV file.

        Returns:
            The loaded pandas DataFrame with no NaNs.
        '''
        df = pd.read_csv(filename)
        df_filling_missing_values = df.ffill()
        return df_filling_missing_values

    def normalizer(self, data):
        '''
        Fits a standard scaler (zero mean, unit variance) on the data.

        Args:
            data: matrix used to fit the scaler.

        Returns:
            A fitted sklearn StandardScaler.
        '''
        norm_scaler = preprocessing.StandardScaler().fit(data)

        return norm_scaler

    def load_norm_data(self, filename):
        '''
        Loads a CSV of sensor data, splits it by the 'block' column into
        training (blocks 0-3) and testing (block 4), and normalizes the test
        set with the scaler fitted on the training set.

        The CSV is expected to have feature columns whose names contain 'x'
        and a label column named 'y'.

        Args:
            filename: path to the CSV file.

        Returns:
            Tuple (x_train_norm, y_train, x_test_norm, y_test).
        '''
        df_filling_missing_values = self.load_csv(filename)
        x_cols = [col for col in df_filling_missing_values.columns if 'x' in col]
        df_training = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                        0, 1, 2, 3])]
        df_testing = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                       4])]

        x_train = df_training[x_cols].to_numpy()
        y_train = df_training['y'].to_numpy()

        x_test = df_testing[x_cols].to_numpy()
        y_test = df_testing['y'].to_numpy()

        norm_scaler = self.normalizer(x_train)
        x_train_norm = norm_scaler.transform(x_train)
        x_test_norm = norm_scaler.transform(x_test)

        return x_train_norm, y_train, x_test_norm, y_test
    
    def load_data(self, filename):
        '''
        Loads a CSV of sensor data and splits it by the 'block' column into
        training (blocks 0-3) and testing (block 4), without normalization.

        The CSV is expected to have feature columns whose names contain 'x'
        and a label column named 'y'.

        Args:
            filename: path to the CSV file.

        Returns:
            Tuple (x_train, y_train, x_test, y_test).
        '''
        df_filling_missing_values = self.load_csv(filename)
        x_cols = [col for col in df_filling_missing_values.columns if 'x' in col]
        df_training = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                        0, 1, 2, 3])]
        df_testing = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                       4])]

        x_train = df_training[x_cols].to_numpy()
        y_train = df_training['y'].to_numpy()

        x_test = df_testing[x_cols].to_numpy()
        y_test = df_testing['y'].to_numpy()

        return x_train, y_train, x_test, y_test

    def gen_data(self, X, y, num_folds=5):
        '''
        Generates time-ordered cross-validation folds. Because the samples are
        a time-ordered list, the data is not shuffled; each fold uses one
        contiguous chunk as validation and the rest as training.

        Args:
            X: data matrix of shape (num_samples, num_features).
            y: label vector of shape (num_samples,).
            num_folds: number of folds. len(y) must be divisible by num_folds.

        Yields:
            Tuples (training_data, training_label, validation_data,
            validation_label) per fold.
        '''
        assert len(
            y) % num_folds == 0, "number of samples must be divisible by num_folds"
        assert num_folds > 1, "number of folds must be bigger than 1"

        num_samples = int(len(y) / num_folds)
        b = [0, num_samples]
        while b[1] <= len(y):
            validation_data = X[b[0]:b[1],]
            validation_label = y[b[0]:b[1]]
            training_data = np.concatenate((X[0:b[0], :], X[b[1]:,]))
            training_label = np.concatenate((y[0:b[0]], y[b[1]:]))
            b[0] += num_samples
            b[1] += num_samples
            yield training_data, training_label, validation_data, validation_label

    def input_data(self, X_train, feature_name, X_test=[], num_feat=128):
        '''
        Applies a feature transform to the training data (and test data, if given).

        feature_name 'raw': returns the data unchanged.
        feature_name 'pca': returns num_feat principal components. Pass in
            normalized data when using PCA.
        feature_name 'eng': returns engineered audio-style features.
        feature_name 'nmf': returns num_feat NMF components fitted on the
            training data and applied to both sets.

        Args:
            X_train: training matrix.
            feature_name: one of 'raw', 'pca', 'eng', 'nmf'.
            X_test: test matrix, or an empty list if unavailable.
            num_feat: number of features for 'pca' and 'nmf'.

        Returns:
            Tuple (Xf_train, Xf_test) of transformed matrices.
        '''
        if feature_name == 'pca':
            pca = self.project_pca(X_train, dim=num_feat)
            Xf_train = pca.transform(X_train)  # eigen feature
            if len(X_test):
                Xf_test = pca.transform(X_test)
            else:
                Xf_test = X_test

            return Xf_train, Xf_test

        elif feature_name == 'eng':
            feat_train = self.feature_engr(X_train)
            feat_test = self.feature_engr(X_test)
            return feat_train, feat_test
        elif feature_name == 'nmf':
            nmf = self.fit_nmf(X_train, dim=num_feat)
            feat_train = self.project_nmf(nmf, X_train)
            feat_test = self.project_nmf(nmf, X_test)
            return feat_train, feat_test
            
        else:
            return X_train, X_test

    def viz_frames(self, X, y, num_frames=32):
        '''
        Visualizes the data as an animated series of frames: the mean sensor
        reading per frame alongside its label.

        Args:
            X: data matrix of shape (num_samples, num_features).
            y: label vector.
            num_frames: number of samples shown per frame, 50% overlap.
        '''
        b = [0, num_frames]
        while b[1] <= len(X):
            fig, axs = plt.subplots(2, 1, figsize=(5, 10))
            axs[0].plot(np.mean(X[b[0]:b[1], :],axis=1))
            axs[0].set_title('50% overlap')
            axs[0].set_xlabel('Time Frames')
            axs[0].set_ylabel('Avg over sensor')
            axs[1].plot(y[b[0]:b[1]])
            axs[1].set_xlabel('Time Frames')
            axs[1].set_ylabel('Features')
            axs[1].set_title('Label')
            axs[1].set_ylim([-0.1, 1.5])
            plt.show()
            b[0] += int(num_frames / 2)
            b[1] += int(num_frames / 2)

            clear_output(wait=True)

    def viz_raw(self, X, y):
        '''
        Plots each sample's windowed mean/std over sensors plus its spectral
        peaks, colored by label.

        Args:
            X: data matrix of shape (num_samples, num_features).
            y: label vector.
        '''
        for idx, x in enumerate(X):
            label = y[idx]
            plt.figure(figsize=(25, 5), dpi=100)

            if label == 1:
                plt.plot(self.stat_window(x)[0], 'r')
                plt.plot(self.stat_window(x)[1], 'g')
            else:
                plt.plot(self.stat_window(x)[0])
                plt.plot(self.stat_window(x)[1])

            peaks_idx, _ = find_peaks(x, height=3)

            plt.title(
                "frame {} peak sensor {} label {}".format(
                    idx, peaks_idx, label))
            plt.ylim([-1, 5])
            plt.legend(['mean', 'std'])
            plt.xlabel('sensors')
            plt.show()
            clear_output(wait=True)

    def viz_projected_data(self, filename, projection='pca', mask=[True]*1024):
        '''
        Loads a CSV, normalizes it, projects it to 3-D, and shows an
        interactive scatter plot colored by label.

        Args:
            filename: path to the CSV file with feature columns and a 'y'
                label column.
            projection: one of 'pca', 'umap', 'tsne', 'nmf'.
            mask: boolean mask selecting which sensor columns to use.

        Returns:
            The projected data matrix of shape (num_samples, 3).
        '''
        df_filling_missing_values = self.load_csv(filename)
        x_cols = [col for col in df_filling_missing_values.columns if 'x' in col]
        zero_data = df_filling_missing_values[df_filling_missing_values['y'] == 0]
        one_data = df_filling_missing_values[df_filling_missing_values['y'] == 1]

        times_series_data_0 = np.array([x[mask] for x in list(zero_data[x_cols].to_numpy())])
        times_series_data_1 = np.array([x[mask] for x in list(one_data[x_cols].to_numpy())])
        
        print(times_series_data_0.shape)
        data = np.concatenate((times_series_data_0, times_series_data_1))

        labels0 = np.zeros((1, len(times_series_data_0)))[0]
        labels1 = np.zeros((1, len(times_series_data_1)))[0] + 1
        labels = np.concatenate((labels0, labels1))

        c = list(zip(data, labels))
        np.random.shuffle(c)
        data, labels = zip(*c)
        data = np.array(data)

        norm_scaler = self.normalizer(data)
        data_norm = norm_scaler.transform(data)
        if projection == 'pca':
            projection = self.project_pca(data_norm, dim=3)
            data_transformation = projection.transform(data_norm)
            self.scatter_plot(data_transformation, labels)
        elif projection == 'umap':
            data_transformation = self.project_umap(data_norm, dim=3)
            self.scatter_plot(data_transformation, labels)
        elif projection == 'tsne':
            data_transformation = self.project_tsne(data_norm, dim=3)
            self.scatter_plot(data_transformation, labels)
        elif projection == 'nmf':
            nmf = self.fit_nmf(data, dim=3)
            data_transformation = self.project_nmf(nmf, data)
            self.scatter_plot(data_transformation, labels)

        return data_transformation
