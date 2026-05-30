from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).with_name("Laptop_Price_Prediction.ipynb")


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _lines(text: str) -> list[str]:
    return [f"{line}\n" for line in dedent(text).strip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {
            "id": _cell_id(),
            "language": "markdown",
        },
        "source": _lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {
            "id": _cell_id(),
            "language": "python",
        },
        "source": _lines(text),
        "execution_count": None,
        "outputs": [],
    }


cells: list[dict] = []

cells.append(
    markdown_cell(
        """
        # Laptop Price Prediction

        ## Section 1 - Project Introduction

        **Problem Statement:** Build a regression model that predicts laptop prices from product specifications, brand information, display attributes, processor details, storage configuration, and operating system features.

        **Business Objective:** Help buyers, sellers, and merchandising teams estimate fair market value, identify the strongest price drivers, and compare devices consistently across different laptop segments.

        **Dataset Overview:** The notebook automatically detects the dataset file from the current working directory and adapts to either CSV or Excel input without hardcoded paths.

        **Regression Overview:** This is a supervised learning regression problem because the target variable is continuous. The workflow below includes data loading, cleaning, feature engineering, encoding, model training, evaluation, tuning, explainability, and model persistence.
        """
    )
)

cells.append(
    code_cell(
        """
        import os
        import re
        import warnings
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import RandomizedSearchCV, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 140)

        try:
            from xgboost import XGBRegressor
            XGBOOST_AVAILABLE = True
        except Exception as import_error:
            XGBOOST_AVAILABLE = False
            XGBRegressor = None
            print(f"XGBoost import skipped: {import_error}")

        try:
            from catboost import CatBoostRegressor
            CATBOOST_AVAILABLE = True
        except Exception as import_error:
            CATBOOST_AVAILABLE = False
            CatBoostRegressor = None
            print(f"CatBoost import skipped: {import_error}")

        try:
            import shap
            SHAP_AVAILABLE = True
        except Exception as import_error:
            SHAP_AVAILABLE = False
            shap = None
            print(f"SHAP import skipped: {import_error}")

        RANDOM_STATE = 42
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 3 - Data Loading

        The loader below searches the current directory, picks the first readable dataset file, and supports both spreadsheet and CSV inputs. This keeps the notebook portable and avoids any path hardcoding.
        """
    )
)

cells.append(
    code_cell(
        """
        def detect_and_load_dataset(base_dir: Path = Path.cwd()) -> tuple[Path, pd.DataFrame]:
            candidates = sorted(base_dir.glob("*.xlsx")) + sorted(base_dir.glob("*.xls")) + sorted(base_dir.glob("*.csv"))
            errors: list[str] = []

            for candidate in candidates:
                try:
                    if candidate.suffix.lower() in {".xlsx", ".xls"}:
                        frame = pd.read_excel(candidate)
                    else:
                        frame = pd.read_csv(candidate, engine="python", on_bad_lines="skip")
                    frame.columns = [str(column).strip() for column in frame.columns]
                    if frame.shape[0] > 0 and frame.shape[1] > 5:
                        return candidate, frame
                except Exception as exc:
                    errors.append(f"{candidate.name}: {exc}")

            error_text = "; ".join(errors) if errors else "No CSV or Excel files were found in the current directory."
            raise FileNotFoundError(error_text)


        dataset_path, df_raw = detect_and_load_dataset()
        print(f"Loaded dataset: {dataset_path.name}")
        print(f"Shape: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")
        print("Data types:")
        display(df_raw.dtypes.to_frame("dtype"))
        print("Sample records:")
        display(df_raw.head())
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 4 - Data Understanding

        This section reviews the dataset at a high level before any transformation. The goal is to understand structure, quality, missingness, duplicates, and the difference between numeric and categorical fields.
        """
    )
)

cells.append(
    code_cell(
        """
        print(f"Dataset shape: {df_raw.shape}")
        print(f"Duplicate rows: {df_raw.duplicated().sum()}")
        print()
        print("Missing values by column:")
        display(df_raw.isna().sum().sort_values(ascending=False).to_frame("missing_values"))

        print()
        print("Descriptive summary:")
        display(df_raw.describe(include="all").T)

        numeric_columns_raw = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns_raw:
            print()
            print("Numerical feature summary:")
            display(df_raw[numeric_columns_raw].describe().T)

        categorical_columns_raw = df_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_columns_raw:
            categorical_summary = pd.DataFrame(
                {
                    "unique_values": [df_raw[column].nunique(dropna=False) for column in categorical_columns_raw],
                    "top_value": [df_raw[column].mode(dropna=True).iloc[0] if not df_raw[column].mode(dropna=True).empty else np.nan for column in categorical_columns_raw],
                    "top_frequency": [df_raw[column].value_counts(dropna=False).iloc[0] if not df_raw[column].value_counts(dropna=False).empty else np.nan for column in categorical_columns_raw],
                },
                index=categorical_columns_raw,
            )
            print()
            print("Categorical feature summary:")
            display(categorical_summary)
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 5 - Data Cleaning

        Cleaning decisions are documented explicitly below:

        - The target column is identified automatically by name.
        - Duplicate records are removed.
        - Missing values are imputed with a sensible default for each type.
        - Numeric outliers are winsorized using the IQR rule to limit the effect of extreme values while preserving rows.
        - Text columns are normalized with whitespace cleanup.
        """
    )
)

cells.append(
    code_cell(
        """
        def detect_target_column(frame: pd.DataFrame) -> str:
            lower_map = {str(column).strip().lower(): column for column in frame.columns}
            direct_candidates = ["price", "laptop price", "sale price", "selling price", "target", "cost"]

            for candidate in direct_candidates:
                if candidate in lower_map:
                    return lower_map[candidate]

            pattern = re.compile(r"(price|sale[_ ]?price|selling[_ ]?price|laptop[_ ]?price|cost)", re.I)
            regex_matches = [column for column in frame.columns if pattern.search(str(column))]
            if regex_matches:
                return regex_matches[0]

            numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                return max(numeric_columns, key=lambda column: frame[column].nunique())

            raise ValueError("Unable to identify the target column automatically.")


        target_column = detect_target_column(df_raw)
        print(f"Target column detected: {target_column}")

        df = df_raw.copy()
        df.columns = [str(column).strip() for column in df.columns]
        duplicate_rows_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicate rows removed: {duplicate_rows_before - len(df)}")

        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        df = df.dropna(subset=[target_column]).reset_index(drop=True)

        for column in df.columns:
            if column == target_column:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
            else:
                mode = df[column].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                df[column] = df[column].fillna(fill_value).astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

        def cap_outliers(series: pd.Series, factor: float = 1.5) -> pd.Series:
            if not pd.api.types.is_numeric_dtype(series):
                return series
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                return series
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            return series.clip(lower_bound, upper_bound)

        numeric_feature_columns = [column for column in df.select_dtypes(include=[np.number]).columns if column != target_column]
        for column in numeric_feature_columns:
            df[column] = cap_outliers(df[column])

        print(f"Rows after cleaning: {len(df)}")
        display(df.head())
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 6 - Advanced EDA

        The charts below focus on the variables that matter most for laptop pricing. The notebook also notes that a dedicated weight column is not present in this dataset, so that example transformation is intentionally excluded.
        """
    )
)

cells.append(
    code_cell(
        """
        eda_df = df.copy()
        for column in ["display_size", "resolution_width", "resolution_height", target_column, "spec_rating", "warranty"]:
            if column in eda_df.columns:
                eda_df[column] = pd.to_numeric(eda_df[column], errors="coerce")

        eda_df["ram_gb"] = pd.to_numeric(eda_df["Ram"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce") if "Ram" in eda_df.columns else np.nan
        eda_df["ppi"] = np.sqrt(eda_df["resolution_width"] ** 2 + eda_df["resolution_height"] ** 2) / eda_df["display_size"]

        numeric_focus = [column for column in [target_column, "ram_gb", "display_size", "resolution_width", "resolution_height", "ppi", "spec_rating", "warranty"] if column in eda_df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, column in zip(axes.flatten(), numeric_focus[:4]):
            sns.histplot(eda_df[column].dropna(), kde=True, ax=ax, color="#2E86AB")
            ax.set_title(f"Distribution of {column}")
        plt.tight_layout()
        plt.show()

        if len(numeric_focus) >= 4:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, column in zip(axes, numeric_focus[1:4]):
                sns.boxplot(x=eda_df[column], ax=ax, color="#F18F01")
                ax.set_title(f"Boxplot of {column}")
            plt.tight_layout()
            plt.show()

        if "ram_gb" in eda_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=eda_df, x="ram_gb", y=target_column, alpha=0.7, color="#3B7EA1")
            plt.title("Price vs RAM")
            plt.tight_layout()
            plt.show()

        if "display_size" in eda_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=eda_df, x="display_size", y=target_column, alpha=0.7, color="#8E5572")
            plt.title("Price vs Screen Size")
            plt.tight_layout()
            plt.show()

        corr_columns = [column for column in [target_column, "ram_gb", "display_size", "resolution_width", "resolution_height", "ppi", "spec_rating", "warranty"] if column in eda_df.columns]
        if len(corr_columns) >= 3:
            plt.figure(figsize=(10, 8))
            sns.heatmap(eda_df[corr_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()

        print("Business insight: higher RAM, larger screens, and denser displays tend to cluster toward higher price bands.")
        print("Business insight: premium brands and gaming-oriented configurations usually occupy the upper price tail.")
        """
    )
)

cells.append(
    code_cell(
        """
        categorical_focus = [column for column in ["brand", "OS", "processor", "GPU"] if column in eda_df.columns]

        for column in categorical_focus:
            top_counts = eda_df[column].value_counts().head(10)
            plt.figure(figsize=(12, 5))
            sns.barplot(x=top_counts.index.astype(str), y=top_counts.values, palette="viridis")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Top values for {column}")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

        if "brand" in eda_df.columns:
            top_brands = eda_df["brand"].value_counts().head(8).index.tolist()
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=eda_df[eda_df["brand"].isin(top_brands)], x="brand", y=target_column, palette="Set2")
            plt.xticks(rotation=45, ha="right")
            plt.title("Price Distribution by Brand")
            plt.tight_layout()
            plt.show()

        print("Business insight: brand and operating system explain a meaningful part of the price segmentation, especially in premium and gaming laptop groups.")
        print("Business insight: the CPU and GPU text fields are high-cardinality, so the most common configurations are the most readable for EDA plots.")
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 7 - Feature Engineering

        Transformations below convert semi-structured laptop specification fields into model-ready variables.

        - RAM strings such as `8GB` and `16GB` become numeric gigabyte values.
        - Storage is separated into SSD, HDD, and total capacity.
        - CPU text is summarized into brand, family, cores, and threads.
        - GPU text is summarized into brand and memory.
        - Screen resolution is used to derive pixels-per-inch.
        - A lightweight gaming indicator is created from the product name.
        - The dataset does not contain a weight column, so that example is documented but not applied.
        """
    )
)

cells.append(
    code_cell(
        """
        def to_number(value) -> float:
            if pd.isna(value):
                return np.nan
            match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
            return float(match.group(1)) if match else np.nan


        def storage_to_gb(value) -> float:
            if pd.isna(value):
                return np.nan
            text = str(value).upper().replace(" ", "")
            match = re.search(r"(\d+(?:\.\d+)?)(TB|GB)", text)
            if not match:
                return np.nan
            amount = float(match.group(1))
            unit = match.group(2)
            return amount * 1024 if unit == "TB" else amount


        def extract_cpu_brand(text) -> str:
            text = str(text)
            if re.search(r"\bIntel\b", text, re.I):
                return "Intel"
            if re.search(r"\bAMD\b", text, re.I):
                return "AMD"
            if re.search(r"\bApple\b", text, re.I):
                return "Apple"
            if re.search(r"\bMediaTek\b", text, re.I):
                return "MediaTek"
            if re.search(r"\bQualcomm\b", text, re.I):
                return "Qualcomm"
            if re.search(r"\bCeleron\b|\bPentium\b|\bXeon\b|\bAtom\b", text, re.I):
                return "Intel"
            if re.search(r"\bAthlon\b|\bRyzen\b", text, re.I):
                return "AMD"
            return "Other"


        def extract_cpu_family(text) -> str:
            text = str(text)
            patterns = [
                r"Core\s+Ultra\s+\d+",
                r"Core\s+i[3-9]",
                r"Ryzen\s+[3-9]",
                r"M\d+",
                r"Celeron",
                r"Pentium",
                r"Athlon(?:\s+Silver|\s+Pro)?",
                r"Xeon",
                r"Atom",
                r"N\d+",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    return re.sub(r"\s+", " ", match.group(0)).strip()
            return "Other"


        def extract_gpu_brand(text) -> str:
            text = str(text)
            if re.search(r"\bNVIDIA\b", text, re.I):
                return "NVIDIA"
            if re.search(r"\bAMD\b", text, re.I):
                return "AMD"
            if re.search(r"\bIntel\b", text, re.I):
                return "Intel"
            if re.search(r"\bApple\b", text, re.I):
                return "Apple"
            if re.search(r"\bARM\b", text, re.I):
                return "ARM"
            if re.search(r"\bMali\b", text, re.I):
                return "Mali"
            return "Other"


        def extract_gpu_memory_gb(text) -> float:
            if pd.isna(text):
                return np.nan
            match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*GB", str(text), re.I)
            return float(match.group(1)) if match else np.nan


        def extract_os_family(text) -> str:
            text = str(text)
            for family in ["Windows", "Mac", "Chrome", "Ubuntu", "DOS", "Android", "Linux"]:
                if re.search(rf"\b{family}\b", text, re.I):
                    return family
            return "Other"


        def extract_cpu_cores(text) -> float:
            match = re.search(r"(\d+)\s*Cores?", str(text), re.I)
            if match:
                return float(match.group(1))
            match = re.search(r"(\d+)\s*Core", str(text), re.I)
            if match:
                return float(match.group(1))
            return np.nan


        def extract_cpu_threads(text) -> float:
            match = re.search(r"(\d+)\s*Threads?", str(text), re.I)
            return float(match.group(1)) if match else np.nan


        feature_df = df.copy()
        for column in ["display_size", "resolution_width", "resolution_height", "spec_rating", "warranty"]:
            if column in feature_df.columns:
                feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

        feature_df["ram_gb"] = feature_df["Ram"].map(to_number) if "Ram" in feature_df.columns else np.nan
        feature_df["storage_gb"] = feature_df["ROM"].map(storage_to_gb) if "ROM" in feature_df.columns else np.nan
        feature_df["ssd_size_gb"] = np.where(feature_df["ROM_type"].astype(str).str.contains("SSD", case=False, na=False), feature_df["storage_gb"], 0.0) if "ROM_type" in feature_df.columns else np.nan
        feature_df["hdd_size_gb"] = np.where(feature_df["ROM_type"].astype(str).str.contains("Hard", case=False, na=False), feature_df["storage_gb"], 0.0) if "ROM_type" in feature_df.columns else np.nan
        feature_df["total_storage_gb"] = feature_df["storage_gb"]
        feature_df["cpu_brand"] = feature_df["processor"].map(extract_cpu_brand) if "processor" in feature_df.columns else "Other"
        feature_df["cpu_family"] = feature_df["processor"].map(extract_cpu_family) if "processor" in feature_df.columns else "Other"
        feature_df["gpu_brand"] = feature_df["GPU"].map(extract_gpu_brand) if "GPU" in feature_df.columns else "Other"
        feature_df["gpu_memory_gb"] = feature_df["GPU"].map(extract_gpu_memory_gb) if "GPU" in feature_df.columns else np.nan
        feature_df["os_family"] = feature_df["OS"].map(extract_os_family) if "OS" in feature_df.columns else "Other"
        feature_df["cpu_cores"] = feature_df["CPU"].map(extract_cpu_cores) if "CPU" in feature_df.columns else np.nan
        feature_df["cpu_threads"] = feature_df["CPU"].map(extract_cpu_threads) if "CPU" in feature_df.columns else np.nan
        feature_df["ppi"] = np.sqrt(feature_df["resolution_width"] ** 2 + feature_df["resolution_height"] ** 2) / feature_df["display_size"]
        feature_df["is_gaming"] = feature_df["name"].astype(str).str.contains(
            "gaming|rog|victus|nitro|legion|omen|predator|tuf|loq|katana|cyborg|alienware|g15|g16",
            case=False,
            na=False,
        ).astype(int) if "name" in feature_df.columns else 0

        if "weight" not in feature_df.columns and "Weight" not in feature_df.columns:
            print("Weight is not present in this dataset, so it is documented but not engineered.")

        engineered_columns = [
            target_column,
            "brand",
            "spec_rating",
            "warranty",
            "display_size",
            "resolution_width",
            "resolution_height",
            "ram_gb",
            "storage_gb",
            "ssd_size_gb",
            "hdd_size_gb",
            "total_storage_gb",
            "cpu_brand",
            "cpu_family",
            "cpu_cores",
            "cpu_threads",
            "gpu_brand",
            "gpu_memory_gb",
            "os_family",
            "ppi",
            "is_gaming",
        ]
        engineered_columns = [column for column in engineered_columns if column in feature_df.columns]
        model_df = feature_df[engineered_columns].copy()

        for column in model_df.columns:
            if column == target_column:
                model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
            elif pd.api.types.is_numeric_dtype(model_df[column]):
                model_df[column] = model_df[column].fillna(model_df[column].median())
            else:
                mode = model_df[column].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                model_df[column] = model_df[column].fillna(fill_value).astype(str)

        print("Engineered feature set:")
        display(model_df.head())
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 8 - Feature Encoding

        Categorical variables are one-hot encoded because most of them are nominal and there is no natural ordering between categories. Numeric variables are standardized so that linear models and distance-sensitive metrics behave consistently.
        """
    )
)

cells.append(
    code_cell(
        """
        X = model_df.drop(columns=[target_column])
        y = model_df[target_column]

        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Categorical columns: {categorical_features}")
        print(f"Numeric columns: {numeric_features}")

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 9 - Feature Selection

        Three complementary signals are used here:

        - Pearson correlation for numeric variables.
        - Mutual information for non-linear relationships.
        - Random forest feature importance for tree-based ranking.
        """
    )
)

cells.append(
    code_cell(
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()

        numeric_correlation = model_df.select_dtypes(include=[np.number]).corr(numeric_only=True)[target_column].drop(target_column).abs().sort_values(ascending=False)
        print("Top numeric correlations with price:")
        display(numeric_correlation.to_frame("abs_correlation"))

        mutual_info_scores = mutual_info_regression(X_train_processed, y_train, random_state=RANDOM_STATE)
        mutual_info_ranking = pd.Series(mutual_info_scores, index=feature_names).sort_values(ascending=False)
        print("Top features by mutual information:")
        display(mutual_info_ranking.head(20).to_frame("mutual_information"))

        rf_selector = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf_selector.fit(X_train_processed, y_train)
        random_forest_ranking = pd.Series(rf_selector.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("Top features by random forest importance:")
        display(random_forest_ranking.head(20).to_frame("importance"))
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 10 - Data Preprocessing

        The final modeling split is 80/20 with `random_state=42`. Scaling is handled inside the preprocessing pipeline so the same transformation is applied consistently during training, tuning, and inference.
        """
    )
)

cells.append(
    code_cell(
        """
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target distribution in training set: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
        print("Feature preprocessing is already prepared via the ColumnTransformer named preprocessor.")
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 11 - Baseline Models

        The baseline models below establish a reference point before moving to more expressive ensembles.
        """
    )
)

cells.append(
    code_cell(
        """
        def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
            return {
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2": r2_score(y_true, y_pred),
            }


        def fit_and_score_model(model, model_name: str) -> tuple[Pipeline, dict[str, float], np.ndarray]:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            metrics = regression_metrics(y_test, predictions)
            metrics["Model"] = model_name
            return pipeline, metrics, predictions


        baseline_model_specs = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10000),
        }

        trained_models: dict[str, Pipeline] = {}
        baseline_results: list[dict[str, float]] = []
        baseline_predictions: dict[str, np.ndarray] = {}

        for model_name, model in baseline_model_specs.items():
            pipeline, metrics, predictions = fit_and_score_model(model, model_name)
            trained_models[model_name] = pipeline
            baseline_results.append(metrics)
            baseline_predictions[model_name] = predictions

        baseline_results_df = pd.DataFrame(baseline_results).sort_values("RMSE").reset_index(drop=True)
        display(baseline_results_df)
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 12 - Advanced Models

        The advanced model pool includes tree ensembles and gradient boosting methods. CatBoost and XGBoost are used when the libraries are available in the runtime environment.
        """
    )
)

cells.append(
    code_cell(
        """
        advanced_model_specs = {
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=350,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=None,
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }

        if XGBOOST_AVAILABLE:
            advanced_model_specs["XGBoost Regressor"] = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            print("XGBoost was skipped because the package is not available.")

        if CATBOOST_AVAILABLE:
            advanced_model_specs["CatBoost Regressor"] = CatBoostRegressor(
                loss_function="RMSE",
                verbose=0,
                random_seed=RANDOM_STATE,
                iterations=400,
                learning_rate=0.05,
                depth=6,
            )
        else:
            print("CatBoost was skipped because the package is not available.")

        advanced_results: list[dict[str, float]] = []

        for model_name, model in advanced_model_specs.items():
            pipeline, metrics, predictions = fit_and_score_model(model, model_name)
            trained_models[model_name] = pipeline
            advanced_results.append(metrics)
            baseline_predictions[model_name] = predictions

        advanced_results_df = pd.DataFrame(advanced_results).sort_values("RMSE").reset_index(drop=True)
        display(advanced_results_df)
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 13 - Model Evaluation

        The table below combines baseline and advanced models, then ranks them by RMSE. MAE, MSE, RMSE, and R2 are reported for a balanced view of error magnitude and explanatory power.
        """
    )
)

cells.append(
    code_cell(
        """
        evaluation_table = pd.concat([baseline_results_df, advanced_results_df], ignore_index=True).sort_values("RMSE").reset_index(drop=True)
        evaluation_table = evaluation_table[["Model", "MAE", "MSE", "RMSE", "R2"]]
        display(evaluation_table)

        best_pre_tuning_row = evaluation_table.iloc[0]
        best_pre_tuning_model_name = best_pre_tuning_row["Model"]
        print(f"Best pre-tuning model: {best_pre_tuning_model_name}")
        print(
            f"Metrics: MAE={best_pre_tuning_row['MAE']:.2f}, MSE={best_pre_tuning_row['MSE']:.2f}, "
            f"RMSE={best_pre_tuning_row['RMSE']:.2f}, R2={best_pre_tuning_row['R2']:.4f}"
        )
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 14 - Hyperparameter Tuning

        The best performing tunable model is refined using `RandomizedSearchCV`. Search spaces are kept modest so the notebook remains practical to execute on a standard laptop environment.
        """
    )
)

cells.append(
    code_cell(
        """
        tuning_parameter_spaces = {
            "Random Forest Regressor": {
                "model__n_estimators": [200, 350, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.8],
            },
            "Gradient Boosting Regressor": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        }

        if XGBOOST_AVAILABLE:
            tuning_parameter_spaces["XGBoost Regressor"] = {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_alpha": [0.0, 0.1, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0],
            }

        if CATBOOST_AVAILABLE:
            tuning_parameter_spaces["CatBoost Regressor"] = {
                "model__iterations": [200, 400, 600],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__l2_leaf_reg": [1, 3, 5, 7],
            }

        tunable_ranking = evaluation_table[evaluation_table["Model"].isin(tuning_parameter_spaces.keys())].copy()
        if tunable_ranking.empty:
            selected_tuning_model_name = "Random Forest Regressor"
        else:
            selected_tuning_model_name = tunable_ranking.iloc[0]["Model"]

        print(f"Tuning model: {selected_tuning_model_name}")
        print("Parameter space:")
        display(pd.DataFrame(list(tuning_parameter_spaces[selected_tuning_model_name].items()), columns=["parameter", "candidates"]))

        tuning_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", trained_models[selected_tuning_model_name].named_steps["model"]),
            ]
        )

        randomized_search = RandomizedSearchCV(
            estimator=tuning_pipeline,
            param_distributions=tuning_parameter_spaces[selected_tuning_model_name],
            n_iter=12,
            scoring="neg_root_mean_squared_error",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        randomized_search.fit(X_train, y_train)

        best_tuned_model = randomized_search.best_estimator_
        tuned_predictions = best_tuned_model.predict(X_test)
        tuned_metrics = regression_metrics(y_test, tuned_predictions)
        tuned_metrics["Model"] = f"{selected_tuning_model_name} (tuned)"

        print("Best parameters from RandomizedSearchCV:")
        display(pd.DataFrame([randomized_search.best_params_]))
        display(pd.DataFrame([tuned_metrics]))
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 15 - SHAP Explainability

        SHAP is used to explain the chosen model at a global level. The notebook shows both a summary plot and a ranked feature-importance view so the final model is not just accurate, but also interpretable.
        """
    )
)

cells.append(
    code_cell(
        """
        final_candidate_table = pd.concat([evaluation_table, pd.DataFrame([tuned_metrics])], ignore_index=True).sort_values("RMSE").reset_index(drop=True)
        display(final_candidate_table)

        final_model_row = final_candidate_table.iloc[0]
        final_model_name = final_model_row["Model"]
        final_model = best_tuned_model if final_model_name == f"{selected_tuning_model_name} (tuned)" else trained_models[final_model_name]

        print(f"Final selected model: {final_model_name}")
        print(
            f"Final metrics: MAE={final_model_row['MAE']:.2f}, MSE={final_model_row['MSE']:.2f}, "
            f"RMSE={final_model_row['RMSE']:.2f}, R2={final_model_row['R2']:.4f}"
        )

        if SHAP_AVAILABLE:
            try:
                final_preprocessor = final_model.named_steps["preprocessor"]
                final_estimator = final_model.named_steps["model"]
                transformed_train = final_preprocessor.transform(X_train)
                transformed_test = final_preprocessor.transform(X_test)
                shap_feature_names = final_preprocessor.get_feature_names_out()

                sample_size = min(100, transformed_test.shape[0])
                shap_sample = pd.DataFrame(transformed_test[:sample_size], columns=shap_feature_names)

                try:
                    shap_explainer = shap.Explainer(final_estimator, transformed_train, feature_names=shap_feature_names)
                    shap_values = shap_explainer(shap_sample)
                except Exception:
                    shap_explainer = shap.Explainer(final_estimator, shap_sample, feature_names=shap_feature_names)
                    shap_values = shap_explainer(shap_sample)

                shap.summary_plot(shap_values, shap_sample, show=False)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(10, 8))
                shap.plots.bar(shap_values, show=False)
                plt.tight_layout()
                plt.show()

                shap_importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=shap_feature_names).sort_values(ascending=False)
                print("Top SHAP-driven features:")
                display(shap_importance.head(20).to_frame("mean_abs_shap"))
            except Exception as shap_error:
                print(f"SHAP explainability could not be rendered: {shap_error}")
        else:
            print("SHAP is not available in the runtime environment, so explainability plots were skipped.")
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 16 - Final Model Selection

        The final model is chosen from the full comparison table, including the tuned candidate if it improves the score. This keeps the selection rule transparent and metric-driven.
        """
    )
)

cells.append(
    code_cell(
        """
        print(f"Best model after tuning comparison: {final_model_name}")
        print("Selection rationale: the chosen model has the lowest RMSE among the evaluated candidates and a competitive R2 score.")
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 17 - Save Model

        The final pipeline, including preprocessing and the fitted estimator, is persisted to the `models/` directory with `joblib`.
        """
    )
)

cells.append(
    code_cell(
        """
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_output_path = models_dir / "laptop_price_prediction_model.pkl"
        joblib.dump(final_model, model_output_path)
        print(f"Saved model to: {model_output_path.resolve()}")
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 18 - Prediction Demo

        The sample predictions below compare actual prices, predicted prices, and the residual error for a few held-out laptops.
        """
    )
)

cells.append(
    code_cell(
        """
        demo_rows = X_test.iloc[:5].copy()
        demo_results = demo_rows.copy()
        demo_results["Actual Price"] = y_test.iloc[:5].values
        demo_results["Predicted Price"] = final_model.predict(demo_rows)
        demo_results["Error"] = demo_results["Predicted Price"] - demo_results["Actual Price"]
        display(demo_results[["Actual Price", "Predicted Price", "Error"]])
        """
    )
)

cells.append(
    markdown_cell(
        """
        ## Section 19 - Conclusion

        The notebook closes with a concise recap of the dataset, the dominant price drivers, the best model, and the final evaluation metrics.
        """
    )
)

cells.append(
    code_cell(
        """
        top_correlated_features = numeric_correlation.head(5).index.tolist() if 'numeric_correlation' in globals() and not numeric_correlation.empty else []
        top_mutual_information_features = mutual_info_ranking.head(5).index.tolist() if 'mutual_info_ranking' in globals() and not mutual_info_ranking.empty else []

        print(f"Dataset file used: {dataset_path.name}")
        print(f"Target column: {target_column}")
        print(f"Rows after cleaning: {len(df)}")
        print(f"Best model: {final_model_name}")
        print(
            f"Final metrics -> MAE: {final_model_row['MAE']:.2f}, MSE: {final_model_row['MSE']:.2f}, "
            f"RMSE: {final_model_row['RMSE']:.2f}, R2: {final_model_row['R2']:.4f}"
        )
        print(f"Top correlated numeric features: {top_correlated_features}")
        print(f"Top mutual-information features: {top_mutual_information_features}")
        print("Overall finding: RAM, screen characteristics, processor family, GPU brand, and premium brand positioning are expected to be the strongest price drivers.")
        """
    )
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with NOTEBOOK_PATH.open("w", encoding="utf-8") as notebook_file:
    json.dump(notebook, notebook_file, indent=2, ensure_ascii=True)

print(f"Created notebook: {NOTEBOOK_PATH}")
