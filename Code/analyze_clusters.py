import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


columns = [
    "Anticoagulant.therapy", "Antiplatelet.therapy", "GCS.init",
    "GCS.motor.init", "Pupil.anomaly.ph", "Osmotherapy.Anomaly",
    "Cardiac.arrest.ph", "SBP.ph", "DBP.ph", "HR.ph", "SBP.ph.min",
    "DBP.ph.min", "HR.ph.max", "Cristalloid.volume", "Colloid.volume",
    "HemoCue.init", "Delta.hemoCue", "Vasopressor.therapy", "SpO2.ph.min",
    "Medcare.time.ph", "GCS", "GCS.motor", "Pupil.anomaly", "TCD.PI.max",
    "FiO2", "Neurosurgery.day0", "IGS.II", "TBI", "Osmotherapy", "IICP", "EVD",
    "Decompressive.craniectomy", "AIS.head", "AIS.face", "AIS.external", "ISS",
    "Shock.index.ph", "Delta.shock.index"
]

df = pd.read_csv("data_clustered.csv", index_col = 0)
data = df[columns]
clusters = df[["25_clusters"]]

extent = data.max() - data.min()


def analyze_cluster(c):
    print("cluster ", c)
    cluster = data.loc[clusters["15_clusters"] == c]
    means = cluster.mean()
    effect = (means - data.mean()).abs() / extent
    # print(effect)
    print(data.mean().loc[effect > 0.15])


analyze_cluster(1)
analyze_cluster(9)
