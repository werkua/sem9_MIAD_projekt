import pandas as pd
#import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from astropy.coordinates import cartesian_to_spherical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ML_output_analysis import run_analysis

BASE_DIR = "ML_Ready_Data"

# files = ["Data/atm_muon.h5",
#          "Data/atm_neutrino_classA.h5",
#          "Data/atm_neutrino_classB.h5",
#          "Data/atm_neutrino_classC.h5",
#          "Data/atm_neutrino_classD.h5",
#          "Data/atm_neutrino_classE.h5",
#          "Data/atm_neutrino_classF.h5",
#          "Data/atm_neutrino_classG.h5",
#          "Data/atm_neutrino_classH.h5"
#         ]

# files for tracks: classB classF
# dfT = pd.read_hdf(f"{BASE_DIR}/atm_neutrino_classB.h5")
# dfF = pd.read_hdf(files[6])

# for showers: classA classC classE classG
# dfS = pd.read_hdf(f"{BASE_DIR}/atm_neutrino_classA.h5")
# dfC = pd.read_hdf(files[3])
# dfE = pd.read_hdf(files[5])
# dfG = pd.read_hdf(files[7])

df = pd.read_csv(f"{BASE_DIR}/dataset_shower_balanced.csv")
dfS = df.sample(frac=1)
dfs = dfS.head(10000)
# dff = pd.read_csv(f"{BASE_DIR}/dataset_track_balanced.csv")
dff = pd.read_hdf(f"Data/atm_muon.h5")
class1 = [0]*len(dff)
dff["class_label"] = class1
dfT = dff.sample(frac=1)
dft = dfT.head(10000)

# # tracks: 1
# class1 = [1]*len(dfT)
# dfT["label"] = class1
# # showers: 0
# class0 = [0]*len(dfS)
# dfS["label"] = class0

dfst = pd.concat([dfs, dft])
xy = dfst.sample(frac=1).reset_index(drop=True)
print(xy)

dirR, dirTheta, dirPhi = cartesian_to_spherical(xy["dir_x"], xy["dir_y"], xy["dir_z"])
xy["dir_theta"] = dirTheta
xy["dir_phi"] = dirPhi

posR, posTheta, posPhi = cartesian_to_spherical(xy["pos_x"], xy["pos_y"], xy["pos_z"])
xy["pos_R"] = posR
xy["pos_theta"] = posTheta
xy["pos_phi"] = posPhi
# # dirR = np.sqrt((xy["dir_x"])**2 + (xy["dir_y"])**2 + (xy["dir_z"])**2)
# # xy["dir_r"] = dirR
# # plot dirR 3D plot
# # plt.figure()
# # ax = plt.axes(projection='3d')
# # ax.scatter3D(xy["dir_x"], xy["dir_y"], dirR, c=dirR, cmap='viridis', s=1)
# # ax.set_xlabel('dir_x')
# # ax.set_ylabel('dir_y')
# # ax.set_zlabel('dirR')
# # plt.title('3D Scatter Plot of Direction Vectors Colored by Magnitude')
# # plt.savefig("3d_dir_vectors.pdf")
# # plt.show()
# # plt.close()

x = xy.drop(columns=["pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z",'class_label'])
print(x)
print("\n")
y = xy[['class_label']].squeeze()
print(y)
print("\n")


#------------------------------------------------------
# ML opt 10000
#------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_cl = RandomForestClassifier(random_state=42, criterion = "entropy")

param_grid = {
    'n_estimators': [1, 2, 3],
    'max_depth': [7],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
}

scorings = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}

grid = GridSearchCV(
    estimator=rf_cl,
    param_grid=param_grid,
    cv=5,
    scoring=scorings,
    refit="accuracy",
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(grid.best_params_)
print(best_model)
joblib.dump(best_model, f"{BASE_DIR}/best2_rf_model.joblib")

results = pd.DataFrame(grid.cv_results_)

cols = [
    "mean_test_accuracy",
    "mean_test_f1",
    "mean_test_precision",
    "mean_test_recall",
    "rank_test_f1",
]

# print(results[cols].sort_values("rank_test_f1"))
print(results[cols])

# best_model = joblib.load(BASE_DIR/"best2_rf_model.joblib")

y_pred = best_model.predict(X_test)

print("accuracy :", accuracy_score(y_test, y_pred))
print("f1       :", f1_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall   :", recall_score(y_test, y_pred))



output_dir = "output_dziecko3"
input_dir = "input_from_Jezusek2"


import os
os.makedirs(input_dir, exist_ok=True)


pred_df = xy.loc[X_test.index].copy()
pred_df["true_label"] = y_test.values
pred_df["pred_label"] = y_pred


# prawdopodobieństwo klasy 1 (Tor)
pred_df["pred_proba"] = best_model.predict_proba(X_test)[:, 1]


pred_path = f"{input_dir}/predictions.csv"
pred_df.to_csv(pred_path, index=False)


# Run analysis form library


run_analysis(
pred_file=pred_path,
output_dir=output_dir,
feat_file=None, # albo np. "input_from_Jezusek2/feature_importance.csv"
class_names=("Kaskada", "Tor")
)
# # ------------------------------------------------------
# # ML total
# # ------------------------------------------------------

# dfST = pd.concat([dfS, dfT])
# XY = dfST.sample(frac=1).reset_index(drop=True)

# dirR, dirTheta, dirPhi = cartesian_to_spherical(XY["dir_x"], XY["dir_y"], XY["dir_z"])
# XY["dir_theta"] = dirTheta
# XY["dir_phi"] = dirPhi

# posR, posTheta, posPhi = cartesian_to_spherical(XY["pos_x"], XY["pos_y"], XY["pos_z"])
# XY["pos_R"] = posR
# XY["pos_theta"] = posTheta
# XY["pos_phi"] = posPhi

# X = XY.drop(columns=["dir_x", "dir_y", "dir_z",'label'])
# Y = XY[['label']].squeeze()

# XX_train, XX_test, yy_train, yy_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# yy_pred = opt.predict(XX_test)

# accuracy = accuracy_score(yy_test, yy_pred)
# classification_rep = classification_report(yy_test, yy_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rep)