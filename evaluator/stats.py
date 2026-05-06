from __future__ import annotations

from .model import EPSILON, OUTLIER_Z_THRESHOLD, ErrorStats, MetricsComparison, RegressionMetrics


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Empty dataset: cannot compute mean.")
    return sum(values) / len(values)


def compute_errors(prices: list[float], predictions: list[float]) -> ErrorStats:
    if not prices:
        raise ValueError("Empty dataset: cannot compute metrics.")
    if len(prices) != len(predictions):
        raise ValueError("Mismatched prediction and target sizes.")

    residuals: list[float] = []
    abs_error_sum = 0.0
    sq_error_sum = 0.0
    max_abs_error = 0.0

    for actual, predicted in zip(prices, predictions):
        residual = predicted - actual
        abs_residual = abs(residual)
        sq_residual = residual * residual

        residuals.append(residual)
        abs_error_sum += abs_residual
        sq_error_sum += sq_residual
        if abs_residual > max_abs_error:
            max_abs_error = abs_residual

    return ErrorStats(
        residuals=tuple(residuals),
        abs_error_sum=abs_error_sum,
        sq_error_sum=sq_error_sum,
        max_abs_error=max_abs_error,
    )


def compute_variance_stats(residuals: tuple[float, ...], sq_error_sum: float) -> tuple[float, float, int]:
    sample_count = len(residuals)
    if sample_count == 0:
        raise ValueError("Cannot compute variance stats on empty residuals.")

    mean_error = sum(residuals) / sample_count
    error_variance = (sq_error_sum / sample_count) - (mean_error * mean_error)
    if error_variance < 0.0:
        error_variance = 0.0
    error_std = error_variance**0.5

    outlier_count = 0
    if error_std > 0.0:
        for residual in residuals:
            z_score = (residual - mean_error) / error_std
            if abs(z_score) > OUTLIER_Z_THRESHOLD:
                outlier_count += 1

    return mean_error, error_std, outlier_count


def compute_r2(prices: list[float], sq_error_sum: float) -> tuple[float | None, bool]:
    if not prices:
        raise ValueError("Empty dataset: cannot compute R2.")

    y_mean = mean(prices)
    ss_tot = 0.0
    for value in prices:
        centered = value - y_mean
        ss_tot += centered * centered

    if ss_tot < EPSILON:
        return None, False
    return 1.0 - (sq_error_sum / ss_tot), True


def compute_metrics(prices: list[float], predictions: list[float]) -> RegressionMetrics:
    errors = compute_errors(prices, predictions)
    sample_count = len(prices)
    mse = errors.sq_error_sum / sample_count
    mean_error, error_std, outlier_count = compute_variance_stats(
        errors.residuals, errors.sq_error_sum
    )
    r2, r2_defined = compute_r2(prices, errors.sq_error_sum)

    return RegressionMetrics(
        mae=errors.abs_error_sum / sample_count,
        mse=mse,
        rmse=mse**0.5,
        r2=r2,
        r2_defined=r2_defined,
        mean_error=mean_error,
        error_std=error_std,
        max_abs_error=errors.max_abs_error,
        outlier_count=outlier_count,
    )


def compare_with_baseline(prices: list[float], model_predictions: list[float]) -> MetricsComparison:
    model_metrics = compute_metrics(prices, model_predictions)

    baseline_value = mean(prices)
    baseline_predictions = [baseline_value] * len(prices)
    baseline_metrics = compute_metrics(prices, baseline_predictions)

    delta_mse = baseline_metrics.mse - model_metrics.mse

    signal_to_noise_ratio: float | None
    if model_metrics.mse <= EPSILON:
        signal_to_noise_ratio = None
    else:
        signal_to_noise_ratio = baseline_metrics.mse / model_metrics.mse

    usefulness_score: float | None
    if baseline_metrics.mse <= EPSILON:
        usefulness_score = None
    else:
        usefulness_score = delta_mse / baseline_metrics.mse

    return MetricsComparison(
        model=model_metrics,
        baseline=baseline_metrics,
        delta_mse=delta_mse,
        signal_to_noise_ratio=signal_to_noise_ratio,
        usefulness_score=usefulness_score,
    )


def correlation(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y):
        raise ValueError("Mismatched series sizes for correlation.")
    if not values_x:
        raise ValueError("Empty series for correlation.")

    x_mean = mean(values_x)
    y_mean = mean(values_y)

    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0

    for x_value, y_value in zip(values_x, values_y):
        x_centered = x_value - x_mean
        y_centered = y_value - y_mean
        covariance += x_centered * y_centered
        variance_x += x_centered * x_centered
        variance_y += y_centered * y_centered

    if variance_x < EPSILON or variance_y < EPSILON:
        return None
    return covariance / ((variance_x * variance_y) ** 0.5)
