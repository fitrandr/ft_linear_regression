#!/usr/bin/env ./env/bin/python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def load_dataset(path: Path) -> tuple[list[float], list[float]]:
    """
    Load a CSV dataset containing car mileage and price information.

    The CSV file must contain two columns:
        - km
        - price

    Each row is read using csv.DictReader, which converts
    the CSV rows into dictionaries using the header names
    as keys.

    Example CSV:
        km,price
        24000,15000
        35000,12000

    Example row produced by DictReader:
        {
            "km": "24000",
            "price": "15000"
        }

    The function:
        1. Opens the CSV file
        2. Checks that required columns exist
        3. Converts values to float
        4. Stores them into two separate lists
        5. Returns both lists

    Args:
        path (Path):
            Path to the CSV dataset file.

    Returns:
        tuple[list[float], list[float]]:
            - First list contains mileage values
            - Second list contains price values

    Raises:
        ValueError:
            If the CSV does not contain 'km'
            or 'price' columns.

        ValueError:
            If the dataset is empty.
    """
    mileages: list[float] = []
    prices: list[float] = []

    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if "km" not in reader.fieldnames or "price" not in reader.fieldnames:
            raise ValueError("CSV must contain 'km' and 'price' columns.")
        for row in reader:
            mileages.append(float(row["km"]))
            prices.append(float(row["price"]))

    if not mileages:
        raise ValueError("Dataset is empty.")
    return mileages, prices


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """
    Estimate the price of a car using a linear regression model.

    This function applies the linear regression formula:

    :contentReference[oaicite:0]{index=0}

    Where:
        - x represents the mileage of the car
        - theta0 is the intercept
        - theta1 is the slope (weight)
        - y_hat is the predicted price

    Example:
        mileage = 50000
        theta0 = 20000
        theta1 = -0.1

        estimated_price =
            20000 + (-0.1 * 50000)
            = 15000

    Args:
        mileage (float):
            Car mileage in kilometers.

        theta0 (float):
            Intercept of the regression line.

        theta1 (float):
            Slope coefficient of the regression line.

    Returns:
        float:
            Estimated car price predicted
            by the linear regression model.
    """
    return theta0 + theta1 * mileage


def mean(values: list[float]) -> float:
    """
    Compute the arithmetic mean of a list of numbers.

    The mean is calculated by dividing the sum of all
    values by the total number of elements.

    Formula:

    :contentReference[oaicite:0]{index=0}

    Example:
        values = [2, 4, 6]

        mean =
            (2 + 4 + 6) / 3
            = 4

    Args:
        values (list[float]):
            List of numeric values.

    Returns:
        float:
            Arithmetic mean of the values.
    """
    return sum(values) / len(values)


def std(values: list[float], values_mean: float) -> float:
    """
    Compute the standard deviation of a list of numbers.

    Standard deviation measures how spread out
    the values are around the mean.

    The variance is computed first:

    :contentReference[oaicite:1]{index=1}

    Then the standard deviation is obtained
    by taking the square root of the variance:

    :contentReference[oaicite:2]{index=2}

    Example:
        values = [2, 4, 6]
        mean = 4

        variance =
            ((2 - 4)^2 + (4 - 4)^2 + (6 - 4)^2) / 3
            = 2.666...

        std =
            sqrt(2.666...)
            = 1.632...

    Args:
        values (list[float]):
            List of numeric values.

        values_mean (float):
            Mean of the values.

    Returns:
        float:
            Standard deviation of the dataset.
    """
    variance = sum((value - values_mean) ** 2 for value in values) / len(values)
    return variance**0.5


def train(
    mileages: list[float],
    prices: list[float],
    learning_rate: float,
    iterations: int,
) -> dict[str, float]:
    """
    Train a simple linear regression model using gradient descent.

    The model learns the relationship between car mileage
    and car price using the equation:

    :contentReference[oaicite:0]{index=0}

    Where:
        - x is the mileage
        - theta0 is the intercept
        - theta1 is the slope
        - y_hat is the predicted price

    The training process uses gradient descent to minimize
    the prediction error by iteratively updating theta0
    and theta1.

    Before training, mileage values are normalized to improve
    convergence speed and numerical stability.

    Normalization formula:

    :contentReference[oaicite:1]{index=1}

    Gradient descent update rules:

    :contentReference[oaicite:2]{index=2}

    :contentReference[oaicite:3]{index=3}

    After training, the normalized parameters are converted
    back to the original mileage scale:

    
::contentReference[oaicite:4]{index=4}


    Training steps:
        1. Compute dataset mean and standard deviation
        2. Normalize mileage values
        3. Initialize parameters to zero
        4. Compute prediction errors
        5. Compute gradients
        6. Update theta parameters
        7. Convert normalized parameters back
           to the original scale

    Args:
        mileages (list[float]):
            List of car mileage values.

        prices (list[float]):
            List of corresponding car prices.

        learning_rate (float):
            Step size used during gradient descent.

        iterations (int):
            Number of training iterations.

    Returns:
        dict[str, float]:
            Dictionary containing:
                - theta0
                - theta1
                - normalized_theta0
                - normalized_theta1
                - km_mean
                - km_std
                - learning_rate
                - iterations
                - samples

    Raises:
        ValueError:
            If all mileage values are identical,
            making standard deviation equal to zero.
    """
     m = len(mileages)
    km_mean = mean(mileages)
    km_std = std(mileages, km_mean)
    if km_std == 0:
        raise ValueError("All mileage values are identical; cannot train a slope.")

    norm_mileages = [(km - km_mean) / km_std for km in mileages]

    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        gradient0 = 0.0
        gradient1 = 0.0

        for mileage, price in zip(norm_mileages, prices):
            error = estimate_price(mileage, theta0, theta1) - price
            gradient0 += error
            gradient1 += error * mileage

        tmp_theta0 = theta0 - learning_rate * (gradient0 / m)
        tmp_theta1 = theta1 - learning_rate * (gradient1 / m)

        theta0 = tmp_theta0
        theta1 = tmp_theta1

    # Convert parameters back to raw mileage scale:
    # y = theta0 + theta1 * ((x - mu) / sigma)
    #   = (theta0 - theta1 * mu / sigma) + (theta1 / sigma) * x
    raw_theta1 = theta1 / km_std
    raw_theta0 = theta0 - (theta1 * km_mean / km_std)

    return {
        "theta0": raw_theta0,
        "theta1": raw_theta1,
        "normalized_theta0": theta0,
        "normalized_theta1": theta1,
        "km_mean": km_mean,
        "km_std": km_std,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "samples": m,
    }


def mse(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float
) -> float:
    """
    Compute the Mean Squared Error (MSE) of a linear regression model.

    The MSE measures how far the predicted values are from the real values
    on average. It is commonly used as a loss function in regression.

    Formula:

    :contentReference[oaicite:0]{index=0}

    Where:
        - m is the number of samples
        - y_hat is the predicted value
        - y is the true value

    In this implementation, predictions are computed using:

    :contentReference[oaicite:1]{index=1}

    Example:
        mileages = [1000, 2000]
        prices = [10, 20]
        theta0 = 5
        theta1 = 0.01

        predictions:
            5 + 0.01*1000 = 15
            5 + 0.01*2000 = 25

        squared errors:
            (15 - 10)^2 = 25
            (25 - 20)^2 = 25

        MSE = (25 + 25) / 2 = 25

    Args:
        mileages (list[float]):
            Input feature values (car mileage).

        prices (list[float]):
            True target values (car prices).

        theta0 (float):
            Intercept of the model.

        theta1 (float):
            Slope of the model.

    Returns:
        float:
            Mean Squared Error of the model.
    """
    m = len(mileages)
    return sum((estimate_price(km, theta0, theta1) - price) ** 2 for km, price in zip(mileages, prices)) / m


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training a linear regression model.

    This function defines and parses CLI arguments used to configure
    the training pipeline, including dataset path, output model path,
    learning rate, and number of iterations.

    Example usage:
        python train.py --dataset data.csv --model model.json

    Available arguments:
        --dataset:
            Path to the input CSV dataset.
            Default: data.csv

        --model:
            Path where the trained model will be saved.
            Default: model.json

        --learning-rate:
            Step size used for gradient descent optimization.
            Default: 0.1

        --iterations:
            Number of gradient descent iterations.
            Default: 10000

    Returns:
        argparse.Namespace:
            Object containing all parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train linear regression on data.csv")
    parser.add_argument(
        "--dataset",
        default="data.csv",
        help="Path to CSV dataset (default: data.csv)",
    )
    parser.add_argument(
        "--model",
        default="model.json",
        help="Path to output model file (default: model.json)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Gradient descent learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of training iterations (default: 10000)",
    )
    return parser.parse_args()

def main() -> None:
    """
    Entry point of the training pipeline.

    This function orchestrates the full workflow of the linear regression project:
        1. Parses command-line arguments
        2. Loads the dataset from a CSV file
        3. Trains a linear regression model using gradient descent
        4. Saves the trained model to a JSON file
        5. Evaluates the model using Mean Squared Error (MSE)
        6. Displays training results in the console

    Workflow:
        - Read CLI arguments (dataset path, model path, hyperparameters)
        - Load (mileages, prices) dataset
        - Train model to obtain theta parameters
        - Save model parameters to disk
        - Compute final training error (MSE)
        - Print summary of results

    Output example:
        Training complete on 1000 samples.
        theta0 = 1234.567890
        theta1 = -0.045678
        MSE    = 345.123456
        Model saved to model.json
    """
    args = parse_args()
    dataset_path = Path(args.dataset)
    model_path = Path(args.model)

    mileages, prices = load_dataset(dataset_path)
    model = train(
        mileages,
        prices,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )

    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    current_mse = mse(mileages, prices, model["theta0"], model["theta1"])
    print(f"Training complete on {model['samples']} samples.")
    print(f"theta0 = {model['theta0']:.6f}")
    print(f"theta1 = {model['theta1']:.6f}")
    print(f"MSE    = {current_mse:.6f}")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()