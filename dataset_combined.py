from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import xlrd
import sys

from torchvision.transforms import transforms
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import h5py
import os
import numpy as np
import math
from random import choices
from radar_scenes.sequence import get_training_sequences, get_validation_sequences, Sequence
from radar_scenes.labels import ClassificationLabel
from radar_scenes.evaluation import per_point_predictions_to_json, PredictionFileSchemas
from sklearn.cluster import DBSCAN
from sklearn import preprocessing


path_to_dataset = "F:/Desktop/bayesianmtl_pointnet/dataset"

RADAR_DEFAULT_MOUNTING = {
    1: {"x": 3.663, "y": -0.873, "yaw": -1.48418552},
    2: {"x": 3.86, "y": -0.70, "yaw": -0.436185662},
    3: {"x": 3.86, "y": 0.70, "yaw": 0.436},
    4: {"x": 3.663, "y": 0.873, "yaw": 1.484},
}

RCS_max = 57.1  #
X_max = 80       # meters
Y_max = 20      # meters
V_max = 115      # kph
Az_max = 1.3     # radians


def get_mounting(sensor_id: int, json_path=None) -> dict:
    """
    Returns the sensor mounting positions of a single sensor with id sensor_id.
    The positions and the azimuth angle are given relative to the car coordinate system.
    :param sensor_id: Integer sensor id.
    :param json_path: str, path to the sensor.json file. If not defined, the default mounting positions are used.
    :return: dictionary containing the x and y position of the sensor in car coordinates as well as the yaw angle:
            structure: {"x": x_val, "y": y_val, "yaw": yaw_val}
    """
    if json_path is None:
        return RADAR_DEFAULT_MOUNTING[sensor_id]
    else:
        with open(json_path, "r") as f:
            data = json.load(f)
        radar_name = "radar_{}".format(sensor_id)
        if radar_name in data:
            return data[radar_name]
        else:
            raise KeyError("Radar {} does not exist in the json file {}.".format(radar_name, json_path))


class RadarDataset(data.Dataset):
    def __init__(self,  return_track_ids=False):
        self.return_track_ids = return_track_ids
        sequence_file = os.path.join(path_to_dataset, "data", "sequences.json")
        if not os.path.exists(sequence_file):
            print("Please modify this example so that it contains the correct path to the dataset on your machine.")
        # self.training_sequences = get_training_sequences(sequence_file)
        self.training_sequences = get_validation_sequences(sequence_file)

        self.datapath=[]
        for sequence_name in self.training_sequences[5:6]:
            print(sequence_name)
            sequence = Sequence.from_json(os.path.join(path_to_dataset, "data", sequence_name, "scenes.json"))
            for scene in sequence.scenes():
                self.datapath.append(scene)

    def features_from_radar_data(self, index):
        """
        Generate a feature vector for each detection in radar_data.
        The spatial coordinates as well as the ego-motion compensated Doppler velocity and the RCS value are used.
        :param radar_data: Input data
        :return: numpy array with shape (len(radar_data), 6), contains the feature vector for each point
        """
        x_cat = []
        features_cat = []
        track_cat = []
        vvids_cat = []
        for i in range(4):
            idx = index + i
            if idx > len(self.datapath)-1:
                idx = index - i
            datapath = self.datapath[idx]
            radar_data = datapath.radar_data[:]
            vr_value = radar_data["y_cc"]
            valid_idx = np.where(abs(vr_value) < 10)[0]
            radar_data = radar_data[valid_idx]

            vr_value = radar_data["x_cc"]
            valid_idx = np.where(vr_value < 80)[0]
            radar_data = radar_data[valid_idx]

            vr_value = radar_data["x_cc"]
            valid_idx = np.where(vr_value > 0)[0]
            radar_data = radar_data[valid_idx]  # retrieve the radar data which belong to this scene

            y_true = np.array([ClassificationLabel.label_to_clabel(x) for x in radar_data["label_id"]])  # map labels
            valid_points = y_true != None  # filter invalid points

            y_true = y_true[valid_points]  # keep only valid points
            y_true = [x.value for x in y_true]  # get value of enum type to work with integers
            if len(y_true) == 0:
                y_true = [5]*len(valid_points)
                valid_points =[True]*len(y_true)

            X = np.zeros((len(radar_data[valid_points]), 7))  # construct feature vector
            X[:, 0] = radar_data[valid_points]["x_cc"]
            X[:, 1] = radar_data[valid_points]["y_cc"]
            X[:, 2] = radar_data[valid_points]["vr_compensated"]
            X[:, 3] = radar_data[valid_points]["rcs"]

            X[:, 4] = radar_data[valid_points]["azimuth_sc"]
            X[:, 5] = y_true


            # ------------------------------------------------------------------------------------
            # Output labels with flow estimation only for non-static targets
            # ------------------------------------------------------------------------------------
            sensor_yaw = np.array([get_mounting(s_id)["yaw"] for s_id in radar_data["sensor_id"]])
            angles = radar_data["azimuth_sc"] + sensor_yaw
            vx = radar_data["vr_compensated"] * np.cos(angles)
            vy = radar_data["vr_compensated"] * np.sin(angles)
            detection = X[np.where(np.array(y_true) != 5)]  # flow only non-static targets
            vx = vx[np.where(np.array(y_true) != 5)]
            vy = vy[np.where(np.array(y_true) != 5)]
            dirVector = np.zeros(X.shape[0])
            if detection.shape[0] != 0:
                dirVector[np.where(np.array(y_true) != 5)] = [math.degrees(math.atan2(vx[i], vy[i])) for i in
                                                              range(vy.shape[0])]  # -pi to pi

            X[:, 6] = dirVector
            try:
                x0 = preprocessing.MinMaxScaler().fit_transform((X[:, 0]).reshape(-1,1))
                x1 = preprocessing.MinMaxScaler().fit_transform((X[:, 1]).reshape(-1,1))
                x2 = preprocessing.MinMaxScaler().fit_transform((X[:, 2]).reshape(-1,1))
                x3 = preprocessing.MinMaxScaler().fit_transform((X[:, 3]).reshape(-1,1))
                x4 = preprocessing.MinMaxScaler().fit_transform((X[:, 4]).reshape(-1,1))
                x5 = np.expand_dims(X[:, 5], axis=1)
                x6 = preprocessing.MinMaxScaler().fit_transform((X[:, 6]).reshape(-1,1))

                X = np.concatenate((x0, x1, x2, x3, x4, x5, x6), axis=1)
            except:
                pass
            #print(X.shape)

            track = radar_data[valid_points]["track_id"]
            vvids = radar_data[valid_points]["uuid"]

            features = np.zeros((len(radar_data[valid_points]), 6))  # all features vector
            features[:, 0] = X[:, 0]
            features[:, 1] = X[:, 1]
            features[:, 2] = X[:, 2]
            features[:, 3] = X[:, 3]

            features[:, 4] = X[:, 4]
            features[:, 5] = X[:, 6]


            if i == 0:
                x_cat = X
                features_cat = features
                track_cat.append(track)
                vvids_cat.append(vvids)
            else:
                x_cat = np.concatenate((x_cat, X), axis=0)
                features_cat = np.concatenate((features_cat, features), axis=0)
                track_cat.append(track)
                vvids_cat.append(vvids)

        return x_cat, np.concatenate(track_cat), np.concatenate(vvids_cat), features_cat

    def __getitem__(self, index):
        '''
        y_true = np.array([ClassificationLabel.label_to_clabel(x) for x in radar_data["label_id"]])  # map labels
        valid_points = y_true != None  # filter invalid points
        y_true = y_true[valid_points]  # keep only valid points
        y_true = [x.value for x in y_true]  # get value of enum type to work with integers
        '''
        X, track, vvid, feature = self.features_from_radar_data(index)  # construct feature vector

        oricheck = 'F:/Desktop/ini_trackids'
        orifilename = 'track_id %d' % (index) + '.npy'
        np.save(os.path.join(oricheck, orifilename), track)

        oricheck1 = 'F:/Desktop/ini_labels'
        orifilename1 = 'label %d' % (index) + '.npy'
        np.save(os.path.join(oricheck1, orifilename1), X[:, 5])

        oricheck2 = 'F:/Desktop/ini_vvids'
        orifilename2 = 'vvid %d' % (index) + '.npy'
        np.save(os.path.join(oricheck2, orifilename2), vvid)

        oricheck3 = 'F:/Desktop/ini_features'
        orifilename3 = 'feature %d' % (index) + '.npy'
        np.save(os.path.join(oricheck3, orifilename3), feature)

        choice = np.random.choice(X.shape[0], 256, replace=True)
        X = X[choice, :]
        track = np.array(track)
        track = track[choice]
        vvid = np.array(vvid)
        vvid = vvid[choice]
        feature = feature[choice, :]

        check = 'F:/Desktop/bayesianmtl_pointnet/trackids'
        filename = 'track_ids %d' % (index) + '.npy'
        np.save(os.path.join(check, filename), track)

        check1 = 'F:/Desktop/bayesianmtl_pointnet/vvids'
        filename1 = 'vvid %d' % (index) + '.npy'
        np.save(os.path.join(check1, filename1), vvid)

        if self.return_track_ids:
            return X[:,:4], X[:,4], track
        else:
            return X[:,:4], X[:,5], np.concatenate((X[:,:3], X[:,4:5]), axis=-1), X[:,6], feature
            # np.concatenate((X[:, 3].reshape(1, -1), X[:, 6].reshape(1, -1)), axis=0)

    def __len__(self):
        return len(self.datapath[:])


if __name__ == '__main__':
        d = RadarDataset()
        for i in range(len(d)):
            _, _, ps, cls = d[i]
            print('cls',cls)
            if len(ps) == 0:
                print('empty')