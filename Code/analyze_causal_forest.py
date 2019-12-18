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
    "Decompressive.craniectomy", "AIS.head", "AIS.face", "AIS.external",
    "Shock.index.ph", "Delta.shock.index"
]

df = pd.read_csv("new.csv", index_col = 0)
data = df[columns]

df2 = pd.read_csv("forestpredictions.csv")
predictions = df2[["V1"]]
predictions.drop(predictions.index[0])

clusters = predictions < 0

extent = data.max() - data.min()


def analyze_patients(c):
    print("cluster ", c)
    cluster = data.loc[clusters["V1"] == c]
    means = cluster.mean()
    effect = (means - data.mean()).abs() / extent
    # print(effect)
    threshold = 0.05
    variables = {
        "clusters": cluster.mean().loc[effect > threshold],
        "global": data.mean().loc[effect > threshold],
    }
    print(pd.DataFrame(variables))


analyze_patients(
    1
)  # 1 for patients with negative hte, 0 for patients with positive hte
