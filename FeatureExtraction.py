import numpy as np
import scipy.stats


class TemporalFeatures:
    def __init__(self, joint_coordinates):
        self.joint_coordinates = joint_coordinates
        self.x = joint_coordinates[: :2]
        self.y = joint_coordinates[1:: 2
                 ]

        print ("self y: ")
        print(self.y)

    def min_feature(self, data):
        return np.min(data)

    def max_feature(self, data):
        return np.max(data)

    def mean_feature(self, data):
        return np.mean(data)

    def std_feature(self, data):
        return np.std(data)

    def energy_feature(self, data):
        return np.sum(np.square(data))

    def rms_feature(self, data):
        return np.sqrt(np.mean(np.square(data)))

    def variance_feature(self, data):
        return np.var(data)

    def skewness_feature(self, data):
        return scipy.stats.skew(data)

    def kurtosis_feature(self, data):
        return scipy.stats.kurtosis(data)

    def median_feature(self, data):
        return np.median(data)

    def mode_feature(self, data):
        return scipy.stats.mode(data)[0][0]

    def percentile_feature(self, data, percentile):
        return np.percentile(data, percentile)

    def range_feature(self, data):
        return np.max(data) - np.min(data)

    def extract_features(self):
        features = []

        feature_functions = [self.min_feature, self.max_feature, self.mean_feature, self.std_feature,
                             self.energy_feature, self.rms_feature, self.variance_feature, self.skewness_feature,
                             self.kurtosis_feature, self.median_feature, self.mode_feature,self.range_feature]

        for func in feature_functions:
            x_feature = func(self.x)
            y_feature = func(self.y)
            features.append(x_feature)
            features.append(y_feature)

        return features