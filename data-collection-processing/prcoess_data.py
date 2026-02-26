import pandas as pd
import numpy as np

def process_data():
    data = pd.read_csv("./captures.csv")
    data["estimated_weight_lbs"] = data["estimated_weight_kg"] * 2.20462
    data["weight_error_abs"] = (data["estimated_weight_lbs"] - data["weight"]).abs()
    data["weight_percent_error"] = (data["weight_error_abs"] / data["weight"]) * 100
    weight_error_mean = np.mean(data["weight_error_abs"])
    weight_perecent_error_mean = np.mean(data["weight_percent_error"])
    print("Total mean weight error (lbs) ", weight_error_mean)
    print("Total mean weight perecent error (%) ", weight_perecent_error_mean)  

    # within 10% error 
    within_10 = data["weight_percent_error"] <= 10
    percent_meeting_pw10 = (within_10.sum() / len(data)) * 100
    print("Total within accuracy: ", percent_meeting_pw10)

    # error by sex
    sex_error = (
        data.groupby("sex")["weight_percent_error"]
        .agg(["count", "mean", "median", "std"])
    )
    print("percent error by sex: ", sex_error)

    # largest error 
    largest_errors = data.sort_values(
        by="weight_percent_error",
        ascending=False
    )
    print(largest_errors.head(10)[
        [
            "rgb_path",
            "weight",
            "estimated_weight_lbs",
            "weight_error_abs",
            "weight_percent_error",
            "sex",
            "height",
        ]
    ])

    # splitting it up into weight groups 
    data["weight_group"] = np.where(data["weight"] < 140, "<140 lbs", ">=140 lbs")
    bias_summary = (
        data.groupby("weight_group")["weight_percent_error"]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    print(bias_summary)

    # looking at the top heaviest
    top_heaviest = (
        data.sort_values(by="weight", ascending=False)
        .head(10)
    )
    print(top_heaviest[[
        "weight",
        "estimated_weight_lbs",
        "weight_error_abs",
        "weight_percent_error",
        "sex"
    ]])

    data["height_cm"] = (
        data["height"]
        .astype(str)
        .str.replace("cm", "", regex=False)
        .str.strip()
        .astype(float)
    )
    # height correlation to weight estimates
    median_height = data["height_cm"].median()
    data["height_group"] = np.where(
        data["height_cm"] < median_height,
        "Shorter",
        "Taller"
    )
    height_summary = (
        data.groupby("height_group")["weight_percent_error"]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    print(height_summary)

    # top 10 shortest people
    top_shortest = (
        data.sort_values(by="height_cm", ascending=True)
        .head(10)
    )
    print(top_shortest[[
        "height_cm",
        "sex",
        "weight",
        "estimated_weight_lbs",
        "weight_percent_error"
    ]])

    # lowest weight estimates
    top_lowest_weight = (
        data.sort_values(by="estimated_weight_lbs", ascending=True)
        .head(10)
    )
    print(top_lowest_weight[[
        "height_cm",
        "sex",
        "weight",
        "estimated_weight_lbs",
        "weight_percent_error"
    ]])

    # activity level summary
    # nothing useful with produced data
    activity_summary = (
        data.groupby("activity_level")["weight_percent_error"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    print(activity_summary)

    # activity level summary
    # nothing useful with produced data
    ethnicity_summary = (
        data.groupby("race_ethnicity")["weight_percent_error"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    print(ethnicity_summary)

process_data()
