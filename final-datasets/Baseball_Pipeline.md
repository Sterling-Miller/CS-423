# Baseball Data Pipeline Documentation

## Pipeline Overview

This pipeline preprocesses the SLICED s01e09 Playoffs 1 baseball dataset to prepare it for machine learning modeling. It handles categorical target encoding, outlier detection and treatment, robust feature scaling, and missing value imputation. All column dropping and downsampling are performed **before** the pipeline.

---

## Pipeline Diagram

<img width="853" alt="Baseball_Pipeline" src="https://github.com/user-attachments/assets/0dfd9857-4406-453d-8da1-451b027149b5" />

---

## Step-by-Step Design Choices

### 1. Home Team Target Encoding (`encode_home_team`)
- **Transformer:** `CustomTargetTransformer(col='home_team', smoothing=10)`
- **Design Choice:** Target encoding with smoothing for the `home_team` column.
- **Rationale:** Replaces each team with the smoothed mean of the target, reducing overfitting and handling rare teams.

### 2. Batter Team Target Encoding (`encode_batter_team`)
- **Transformer:** `CustomTargetTransformer(col='batter_team', smoothing=10)`
- **Design Choice:** Target encoding with smoothing for the `batter_team` column.
- **Rationale:** Captures the relationship between batter's team and home run probability.

### 3. Away Team Target Encoding (`encode_away_team`)
- **Transformer:** `CustomTargetTransformer(col='away_team', smoothing=10)`
- **Design Choice:** Target encoding with smoothing for the `away_team` column.
- **Rationale:** Encodes categorical team information in a way that reflects its impact on the target.

### 4. Batter Name Target Encoding (`encode_batter_name`)
- **Transformer:** `CustomTargetTransformer(col='batter_name', smoothing=15)`
- **Design Choice:** Higher smoothing for batter names due to many unique values.
- **Rationale:** Prevents overfitting to rare batters while still capturing skill differences.

### 5. Pitcher Name Target Encoding (`encode_pitcher_name`)
- **Transformer:** `CustomTargetTransformer(col='pitcher_name', smoothing=15)`
- **Design Choice:** Higher smoothing for pitcher names.
- **Rationale:** Similar to batter names, reduces noise from rare pitchers.

### 6. Batted Ball Type Target Encoding (`encode_bb_type`)
- **Transformer:** `CustomTargetTransformer(col='bb_type', smoothing=5)`
- **Design Choice:** Target encoding for batted ball type.
- **Rationale:** Encodes the effect of different batted ball types on home run likelihood.

### 7. Bearing Target Encoding (`encode_bearing`)
- **Transformer:** `CustomTargetTransformer(col='bearing', smoothing=5)`
- **Design Choice:** Target encoding for bearing.
- **Rationale:** Captures the relationship between bearing and home run probability.

### 8. Outlier Treatment for Launch Speed (`tukey_launch_speed`)
- **Transformer:** `CustomTukeyTransformer(target_column='launch_speed', fence='outer')`
- **Design Choice:** Tukey's outer fence for outlier clipping.
- **Rationale:** Removes extreme outliers in launch speed that could skew the model.

### 9. Outlier Treatment for Launch Angle (`tukey_launch_angle`)
- **Transformer:** `CustomTukeyTransformer(target_column='launch_angle', fence='outer')`
- **Design Choice:** Tukey's outer fence for outlier clipping.
- **Rationale:** Ensures extreme launch angles do not distort feature scaling.

### 10. Launch Speed Scaling (`scale_launch_speed`)
- **Transformer:** `CustomRobustTransformer(target_column='launch_speed')`
- **Design Choice:** Robust scaling using median and IQR.
- **Rationale:** Reduces the influence of outliers on launch speed.

### 11. Launch Angle Scaling (`scale_launch_angle`)
- **Transformer:** `CustomRobustTransformer(target_column='launch_angle')`
- **Design Choice:** Robust scaling for launch angle.
- **Rationale:** Handles skewed distributions and outliers.

### 12. Imputation (`impute`)
- **Transformer:** `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice:** KNN imputation with 5 neighbors.
- **Rationale:** Uses relationships between features to estimate missing values, more robust than mean/median imputation.

---

## Pipeline Execution Order Rationale

- **Categorical encoding** is performed first to convert string categories to numeric values.
- **Target encoding** is applied before any numerical transformations, as it requires original categorical values.
- **Outlier treatment** is done before scaling to prevent outliers from affecting scaling parameters.
- **Robust scaling** is applied before imputation so that KNN distances are meaningful.
- **Imputation** is performed last to fill in any missing values using all preprocessed features.

---

## Performance Considerations

- **RobustScaler** is used instead of StandardScaler due to the presence of outliers.
- **KNN imputation** preserves relationships between features and is more flexible than simple imputation.
- **Target encoding with smoothing** is used for categorical features with many levels to prevent overfitting.

---

**Note:**  
All column dropping and downsampling are performed before this pipeline. The pipeline is saved as `final_fully_fitted_pipeline.pkl` for reuse in production and future modeling steps.
