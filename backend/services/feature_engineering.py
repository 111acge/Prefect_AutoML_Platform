"""特征工程扩展服务。

基于数据特征自动选择并应用：
- 异常值截断（IQR）
- 数值缩放（Standard / Robust / MinMax）
- 高基数类别低频项合并
- 时间特征提取与周期编码
- 文本 Embedding（可选，需 sentence-transformers）

所有学习到的统计量都在训练集上 fit，验证/测试集只做 transform。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """可序列化的特征工程器。"""

    def __init__(
        self,
        target_column: str,
        outlier_strategy: str = "iqr",
        scaler_type: str = "auto",
        rare_category_threshold: int = 10,
        datetime_cyclical: bool = True,
        text_embeddings: bool = True,
        text_model_name: str = "paraphrase-MiniLM-L3-v2",
        one_hot_threshold: int = 10,
        target_encoding_threshold: int = 50,
        missing_indicator: bool = True,
        missing_indicator_threshold: float = 0.05,
        correlation_threshold: float = 0.95,
        low_variance_threshold: float = 0.0,
        pca_variance_ratio: Optional[float] = 0.95,
        enable_pca: bool = False,
    ):
        self.target_column = target_column
        self.outlier_strategy = outlier_strategy
        self.scaler_type = scaler_type
        self.rare_category_threshold = rare_category_threshold
        self.datetime_cyclical = datetime_cyclical
        self.text_embeddings = text_embeddings
        self.text_model_name = text_model_name
        self.one_hot_threshold = one_hot_threshold
        self.target_encoding_threshold = target_encoding_threshold
        self.missing_indicator = missing_indicator
        self.missing_indicator_threshold = missing_indicator_threshold
        self.correlation_threshold = correlation_threshold
        self.low_variance_threshold = low_variance_threshold
        self.pca_variance_ratio = pca_variance_ratio
        self.enable_pca = enable_pca

        # 学习到的参数
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols: List[str] = []
        self.text_cols: List[str] = []
        self.scaler: Optional[Any] = None
        self.scaler_columns: List[str] = []
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.rare_category_maps: Dict[str, Dict[Any, Any]] = {}
        self.text_encoder: Optional[Any] = None
        self.one_hot_maps: Dict[str, List[Any]] = {}
        self.target_encodings: Dict[str, Dict[Any, float]] = {}
        self.global_target_mean: Optional[float] = None
        self.missing_indicator_cols: List[str] = []
        self.high_correlation_drop: List[str] = []
        self.low_variance_drop: List[str] = []
        self.pca: Optional[Any] = None
        self.pca_columns: List[str] = []

    def fit(self, df: pd.DataFrame):
        """在训练集上学习所有特征工程参数。"""
        df = df.copy()
        self._identify_columns(df)

        if self.outlier_strategy == "iqr":
            self.outlier_bounds = _learn_iqr_bounds(df, self.numeric_cols)

        chosen_scaler = self.scaler_type
        if chosen_scaler == "auto":
            chosen_scaler = _choose_scaler_type(df, self.numeric_cols, self.outlier_bounds)
        self.scaler = _fit_scaler(df, self.numeric_cols, chosen_scaler)
        self.scaler_columns = [c for c in self.numeric_cols if c in df.columns]

        self.rare_category_maps = _learn_rare_category_maps(
            df, self.categorical_cols, self.rare_category_threshold
        )

        # 类别编码：低基数 One-Hot、高基数 Target Encoding
        self.one_hot_maps, self.target_encodings, self.global_target_mean = _learn_categorical_encodings(
            df,
            self.categorical_cols,
            self.target_column,
            self.one_hot_threshold,
            self.target_encoding_threshold,
        )

        # 缺失指示列
        self.missing_indicator_cols = []
        if self.missing_indicator:
            for col in self.numeric_cols + self.categorical_cols + self.datetime_cols:
                if col in df.columns and df[col].isnull().mean() > self.missing_indicator_threshold:
                    self.missing_indicator_cols.append(col)

        if self.text_embeddings and self.text_cols:
            self.text_encoder = _load_text_encoder(self.text_model_name)
            if self.text_encoder is None:
                logger.warning("sentence-transformers 未安装，跳过文本 Embedding")

        # 条件降维
        feature_df = self._get_feature_df(df)
        self.high_correlation_drop = _learn_high_correlation_drop(
            feature_df, self.correlation_threshold
        )
        self.low_variance_drop = _learn_low_variance_drop(
            feature_df, self.low_variance_threshold
        )
        reduced_df = feature_df.drop(columns=self.high_correlation_drop + self.low_variance_drop, errors="ignore")
        if self.enable_pca and self.pca_variance_ratio is not None and not reduced_df.empty:
            self.pca, self.pca_columns = _learn_pca(reduced_df, self.pca_variance_ratio)

        logger.info(
            f"FeatureEngineer 拟合完成: numeric={len(self.numeric_cols)}, "
            f"categorical={len(self.categorical_cols)}, datetime={len(self.datetime_cols)}, "
            f"text={len(self.text_cols)}, one_hot={len(self.one_hot_maps)}, "
            f"target_encoding={len(self.target_encodings)}, scaler={chosen_scaler}, "
            f"corr_drop={len(self.high_correlation_drop)}, var_drop={len(self.low_variance_drop)}, "
            f"pca={self.pca is not None}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用特征工程变换。"""
        df = df.copy()

        # 1. 异常值截断
        if self.outlier_strategy == "iqr":
            df = _apply_iqr_caps(df, self.outlier_bounds)

        # 2. 数值缩放
        if self.scaler is not None and self.scaler_columns:
            df = _apply_scaler(df, self.scaler, self.scaler_columns)

        # 3. 稀有类别合并
        df = _apply_rare_category_maps(df, self.rare_category_maps)

        # 4. 类别编码
        df = _apply_categorical_encodings(
            df,
            self.one_hot_maps,
            self.target_encodings,
            self.global_target_mean,
        )

        # 5. 缺失指示列
        if self.missing_indicator:
            df = _add_missing_indicators(df, self.missing_indicator_cols)

        # 6. 时间特征
        df = _extract_datetime_features(df, self.datetime_cols, self.datetime_cyclical)

        # 7. 条件降维（仅对数值列）
        df = df.drop(columns=self.high_correlation_drop + self.low_variance_drop, errors="ignore")
        if self.pca is not None and self.pca_columns:
            feature_df = self._get_feature_df(df)
            pca_df = _apply_pca(feature_df, self.pca, self.pca_columns)
            df = pd.concat([df[[self.target_column]], pca_df], axis=1)

        # 5. 文本 Embedding
        if self.text_encoder is not None and self.text_cols:
            df = _apply_text_embeddings(df, self.text_cols, self.text_encoder)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _get_feature_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """获取不含目标列的数值特征 DataFrame（用于降维）。"""
        cols = [c for c in df.columns if c != self.target_column]
        feature_df = df[cols].copy()
        numeric_cols = [
            c for c in cols if pd.api.types.is_numeric_dtype(feature_df[c])
        ]
        feature_df = feature_df[numeric_cols]
        # 填充缺失值，避免 PCA / 相关性计算失败
        return feature_df.fillna(feature_df.median())

    def _identify_columns(self, df: pd.DataFrame):
        """识别各类特征列。"""
        feature_cols = [c for c in df.columns if c != self.target_column]

        self.numeric_cols = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
        ]
        self.categorical_cols = [
            c
            for c in feature_cols
            if pd.api.types.is_object_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        ]
        self.datetime_cols = [
            c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])
        ]

        # 文本列：object 类型且唯一值比例高、平均长度较长
        self.text_cols = []
        for c in self.categorical_cols:
            unique_ratio = df[c].nunique() / max(len(df), 1)
            avg_len = df[c].astype(str).str.len().mean()
            if unique_ratio > 0.5 and avg_len > 15:
                self.text_cols.append(c)
        # 从类别列中排除文本列
        self.categorical_cols = [c for c in self.categorical_cols if c not in self.text_cols]


def _learn_iqr_bounds(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """学习 IQR 截断边界。"""
    bounds = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        bounds[col] = (lower, upper)
    return bounds


def _apply_iqr_caps(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """应用 IQR 截断。"""
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _choose_scaler_type(
    df: pd.DataFrame,
    numeric_cols: List[str],
    outlier_bounds: Dict[str, Tuple[float, float]],
) -> str:
    """根据异常值比例选择缩放器。"""
    if not numeric_cols:
        return "none"

    total = 0
    outlier_count = 0
    for col in numeric_cols:
        if col not in outlier_bounds:
            continue
        series = df[col].dropna()
        lower, upper = outlier_bounds[col]
        total += len(series)
        outlier_count += int((series < lower).sum() + (series > upper).sum())

    outlier_ratio = outlier_count / max(total, 1)
    if outlier_ratio > 0.05:
        return "robust"
    return "standard"


def _fit_scaler(df: pd.DataFrame, numeric_cols: List[str], scaler_type: str):
    """拟合缩放器。"""
    if scaler_type == "none" or not numeric_cols:
        return None

    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return None

    X = df[cols].fillna(df[cols].median()).values
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def _apply_scaler(df: pd.DataFrame, scaler, columns: List[str]) -> pd.DataFrame:
    """应用缩放器。"""
    df = df.copy()
    present_cols = [c for c in columns if c in df.columns]
    if not present_cols:
        return df

    # 保存原始索引，处理缺失值
    original_index = df.index
    fill_values = df[present_cols].median()
    X = df[present_cols].fillna(fill_values).values
    X_scaled = scaler.transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=present_cols, index=original_index)
    for col in present_cols:
        df[col] = scaled_df[col]
    return df


def _learn_rare_category_maps(
    df: pd.DataFrame, categorical_cols: List[str], threshold: int
) -> Dict[str, Dict[Any, Any]]:
    """学习稀有类别映射。"""
    maps = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        mapping = {}
        for val, count in counts.items():
            if count < threshold:
                mapping[val] = "__other__"
            else:
                mapping[val] = val
        maps[col] = mapping
    return maps


def _apply_rare_category_maps(
    df: pd.DataFrame, maps: Dict[str, Dict[Any, Any]]
) -> pd.DataFrame:
    """应用稀有类别映射；未命中映射的取值保持原样。"""
    for col, mapping in maps.items():
        if col in df.columns:
            mapped = df[col].map(mapping)
            # 未命中映射的保持原值，避免把填充后新出现的值误判为 __other__
            df[col] = mapped.where(mapped.notna(), df[col])
    return df


def _extract_datetime_features(
    df: pd.DataFrame, datetime_cols: List[str], cyclical: bool
) -> pd.DataFrame:
    """提取时间特征。"""
    for col in datetime_cols:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dayofweek"] = dt.dt.dayofweek
        df[f"{col}_hour"] = dt.dt.hour

        if cyclical:
            df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    return df


def _load_text_encoder(model_name: str) -> Optional[Any]:
    """加载文本编码器（可选）。"""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as e:
        logger.warning(f"文本编码器加载失败: {e}")
        return None


def _apply_text_embeddings(
    df: pd.DataFrame, text_cols: List[str], encoder
) -> pd.DataFrame:
    """应用文本 Embedding。"""
    for col in text_cols:
        if col not in df.columns:
            continue
        texts = df[col].astype(str).fillna("").tolist()
        try:
            embeddings = encoder.encode(texts, show_progress_bar=False)
            for i in range(embeddings.shape[1]):
                df[f"{col}_emb_{i}"] = embeddings[:, i]
        except Exception as e:
            logger.warning(f"文本 Embedding 生成失败 ({col}): {e}")
    return df



def _learn_categorical_encodings(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_column: str,
    one_hot_threshold: int,
    target_encoding_threshold: int,
) -> tuple:
    """学习类别编码：低基数 One-Hot、高基数 Target Encoding。"""
    one_hot_maps: Dict[str, List[Any]] = {}
    target_encodings: Dict[str, Dict[Any, float]] = {}
    global_target_mean: Optional[float] = None

    if target_column not in df.columns:
        return one_hot_maps, target_encodings, global_target_mean

    y = df[target_column]
    global_target_mean = float(y.mean()) if pd.api.types.is_numeric_dtype(y) else None

    for col in categorical_cols:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= one_hot_threshold:
            one_hot_maps[col] = df[col].dropna().unique().tolist()
        elif n_unique >= target_encoding_threshold and global_target_mean is not None:
            # Target Encoding 仅适用于数值目标（回归/二分类）
            encoding = df.groupby(col)[target_column].mean().to_dict()
            target_encodings[col] = {k: float(v) for k, v in encoding.items()}

    return one_hot_maps, target_encodings, global_target_mean


def _apply_categorical_encodings(
    df: pd.DataFrame,
    one_hot_maps: Dict[str, List[Any]],
    target_encodings: Dict[str, Dict[Any, float]],
    global_target_mean: Optional[float],
) -> pd.DataFrame:
    """应用类别编码。"""
    for col, values in one_hot_maps.items():
        if col not in df.columns:
            continue
        for val in values:
            df[f"{col}_{val}"] = (df[col] == val).astype(int)
        df = df.drop(columns=[col])

    for col, encoding in target_encodings.items():
        if col not in df.columns:
            continue
        df[f"{col}_te"] = df[col].map(encoding)
        if global_target_mean is not None:
            df[f"{col}_te"] = df[f"{col}_te"].fillna(global_target_mean)
        df = df.drop(columns=[col])

    return df


def _add_missing_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """为指定列添加缺失指示列。"""
    for col in cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
    return df



def _learn_high_correlation_drop(df: pd.DataFrame, threshold: float) -> List[str]:
    """学习高相关性剔除列（|r| > threshold 时删除其一）。"""
    if df.shape[1] < 2:
        return []
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for other in high_corr:
            if col not in to_drop and other not in to_drop:
                to_drop.add(other)
    return list(to_drop)


def _learn_low_variance_drop(df: pd.DataFrame, threshold: float) -> List[str]:
    """学习低方差剔除列。"""
    if df.empty:
        return []
    from sklearn.feature_selection import VarianceThreshold

    try:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        return [c for i, c in enumerate(df.columns) if not selector.get_support()[i]]
    except Exception as e:
        logger.warning(f"低方差剔除失败: {e}")
        return []


def _learn_pca(df: pd.DataFrame, variance_ratio: float) -> tuple:
    """学习 PCA，保留指定方差比例。"""
    from sklearn.decomposition import PCA

    n_components = min(df.shape[1], max(1, int(df.shape[1] * 0.5)))
    try:
        pca = PCA(n_components=variance_ratio)
        pca.fit(df)
        return pca, df.columns.tolist()
    except Exception as e:
        logger.warning(f"PCA 学习失败: {e}")
        return None, []


def _apply_pca(df: pd.DataFrame, pca, columns: List[str]) -> pd.DataFrame:
    """应用 PCA 变换。"""
    present_cols = [c for c in columns if c in df.columns]
    if not present_cols:
        return df
    X = df[present_cols].fillna(df[present_cols].median()).values
    X_pca = pca.transform(X)
    pca_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
