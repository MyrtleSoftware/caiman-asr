from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from beartype import beartype
from matplotlib import pyplot as plt


def parse_args() -> Namespace:
    #
    parser = ArgumentParser(
        description="Regression script, this script estimates a linear model "
        "for VRAM usage based on the sum of squared duration and area "
        "(time*tokens) of batches. The coefficients can then be used for 2D "
        "bucketing."
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input files for linear regression",
    )
    parser.add_argument(
        "--use_last_n",
        type=int,
        default=10,
        help="Number of datapoints to use from the end of each file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot predicted vs actual VRAM",
    )

    return parser.parse_args()


@beartype
@dataclass
class DataPoint:
    batch_size: int
    vram: float  # MB
    sum_dur: float
    squ_dur: float
    sum_tok: int
    squ_tok: int
    area: float
    time: int

    @property
    def eff_dur(self) -> float:
        return self.sum_dur / self.squ_dur

    @property
    def eff_tok(self) -> float:
        return self.sum_tok / self.squ_tok

    @property
    def eff_area(self) -> float:
        return self.eff_tok * self.eff_dur


def main(args):
    # === Preprocessing ===

    vars = [
        # "batch_size",
        # "sum_dur",
        "squ_dur",
        # "sum_tok",
        # "squ_tok",
        "area",
        # "time",
        #     "eff_dur",
        #     "eff_tok",
        #     "eff_area",
    ]

    # List to hold a temporary DataFrame for each file/cluster
    df_list = []

    for run_id, file in enumerate(args.inputs):
        datapoints = _read_datapoints(file)

        if len(datapoints) < args.use_last_n:
            continue

        datapoints_to_use = datapoints[-args.use_last_n :]

        if not datapoints_to_use:
            continue

        x_data_file = [[getattr(x, v) for v in vars] for x in datapoints_to_use]
        y_data_file = [x.vram for x in datapoints_to_use]

        temp_df = pd.DataFrame(x_data_file, columns=vars, dtype=np.float64)
        temp_df["VRAM"] = y_data_file
        temp_df["run_id"] = run_id

        df_list.append(temp_df)

    use_clusters = len(df_list) > 1

    df = pd.concat(df_list, ignore_index=True)

    Y_series = df["VRAM"]
    X_df = df[vars]

    # === Standardize ===

    x_mean = X_df.mean()
    x_stds = X_df.std(ddof=1)
    X_standardized = (X_df - x_mean) / x_stds

    X_aug = sm.add_constant(X_standardized)

    # === Fit

    stats_model = sm.OLS(Y_series, X_aug).fit()

    robust_model = (
        stats_model.get_robustcov_results(cov_type="cluster", groups=df["run_id"])
        if use_clusters
        else stats_model
    )

    print(robust_model.summary())

    print("\n--- Unstandardized Coefficients ---")
    params = robust_model.params[1:]
    adjusted_params = params / x_stds
    print(adjusted_params)

    inter = robust_model.params[0] - (params * x_mean / x_stds).sum()
    print("Unstandardized intercept:", inter)

    if args.plot:
        pred_y = robust_model.predict(X_aug)

        plt.scatter(pred_y, Y_series)

        ideal = np.linspace(min(Y_series), max(Y_series), 100)

        plt.plot(ideal, ideal, color="red", linestyle="--", label="Ideal")
        plt.xlabel("Predicted VRAM (MB)")
        plt.ylabel("Actual VRAM (MB)")
        plt.title(f"OLS VRAM model ({len(vars)} variables)")
        plt.legend()
        plt.savefig("out.png")

    print(f"Ratio={adjusted_params['squ_dur'] / adjusted_params['area']:.2f}")


def _line_to_datapoint(n: int, line: str) -> DataPoint:
    splits = line.strip().split(",")

    assert len(splits) == 7, f"Expected 7 values per line, got {len(splits)}: {line}"

    return DataPoint(
        batch_size=int(splits[0]),
        vram=float(splits[1]),
        sum_dur=float(splits[2]),
        squ_dur=float(splits[3]),
        sum_tok=int(splits[4]),
        squ_tok=int(splits[5]),
        area=float(splits[6]),
        time=n,
    )


def _read_datapoints(file: Path) -> list[DataPoint]:
    with file.open("r") as f:
        return [_line_to_datapoint(n, line) for n, line in enumerate(f) if line.strip()]


if __name__ == "__main__":
    main(parse_args())
