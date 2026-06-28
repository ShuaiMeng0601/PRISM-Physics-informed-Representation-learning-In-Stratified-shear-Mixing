import csv
import json
import re

import h5py
import torch
from torch.utils.data import Dataset


DEFAULT_VARIABLES = ["u", "v", "w", "th", "epsilon"]


def parse_input_variables(input_variables):
    if input_variables is None:
        return None
    if isinstance(input_variables, str):
        return [item.strip() for item in input_variables.split(",") if item.strip()]
    return list(input_variables)


def parse_case_parameters(case_id):
    """Parse case_id like KH:Ri016_a05_Re2000."""
    if ":" in case_id:
        family, case_name = case_id.split(":", 1)
    else:
        family, case_name = "unknown", case_id

    match = re.search(r"Ri(\d+)_a(\d+)_Re(\d+)", case_name)
    if match is not None:
        ri = int(match.group(1)) / 100.0
        a = int(match.group(2)) / 10.0
        re_number = int(match.group(3))

        condition = torch.tensor([ri, a, re_number / 1000.0], dtype=torch.float32)
        return family, case_name, condition

    match = re.search(r"Re(\d+)_Ri(\d+)_Pr\d+_A(\d+)", case_name)
    if match is not None:
        re_number = int(match.group(1))
        ri = int(match.group(2)) / 100.0
        a_digits = match.group(3)
        a = int(a_digits) / (10 ** len(a_digits))

        condition = torch.tensor([ri, a, re_number / 1000.0], dtype=torch.float32)
        return family, case_name, condition

    match = re.search(r"Ri(\d+)_sym_Re(\d+)", case_name)
    if match is not None:
        ri = int(match.group(1)) / 100.0
        re_number = int(match.group(2))

        condition = torch.tensor([ri, 0.0, re_number / 1000.0], dtype=torch.float32)
        return family, case_name, condition

    condition = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    return family, case_name, condition


def load_ratio_labels(label_csv):
    """Load pseudo momentum ratio labels.
        current case-level table
        Type, Ri, Re, a, R, R_M, Lambda_unstable
    """
    if label_csv is None:
        return {}

    labels = {}
    with open(label_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            explicit_case_id = row.get("case_id", "").strip()
            if explicit_case_id:
                labels[explicit_case_id] = float(row["R_M"])
                continue

            row_type = row["Type"]
            family = "KH" if "KH" in row_type else "Holmboe"
            ri = int(round(float(row["Ri"]) * 100))
            re_number = int(row["Re"])
            if row["a"].strip().lower() == "sym":
                case_name = f"Ri{ri:03d}_sym_Re{re_number}"
            else:
                a = int(round(float(row["a"]) * 10))
                case_name = f"Ri{ri:03d}_a{a:02d}_Re{re_number}"
            case_id = f"{family}:{case_name}"
            labels[case_id] = float(row["R_M"])

    return labels


class KHHolmboeDataset(Dataset):
    """Read train/val/test data from kh_holmboe_dataset.h5."""

    def __init__(self, h5_path, split="train", label_csv=None, input_variables=None):
        self.h5_path = h5_path
        self.split = split
        self.labels = load_ratio_labels(label_csv)

        self.h5 = h5py.File(h5_path, "r")
        self.x_data = self.h5[f"{split}/X"]
        raw_variables = self.h5.attrs.get("variables")
        if raw_variables is None:
            self.available_variables = DEFAULT_VARIABLES[:self.x_data.shape[1]]
        else:
            if isinstance(raw_variables, bytes):
                raw_variables = raw_variables.decode("utf-8")
            self.available_variables = json.loads(raw_variables)

        requested_variables = parse_input_variables(input_variables)
        self.input_variables = requested_variables or list(self.available_variables)
        self.channel_indices = []
        self.missing_variables = []
        for name in self.input_variables:
            if name in self.available_variables:
                self.channel_indices.append(self.available_variables.index(name))
            else:
                self.channel_indices.append(None)
                self.missing_variables.append(name)

        if "th" in self.input_variables:
            self.theta_index = self.input_variables.index("th")
        elif "theta" in self.input_variables:
            self.theta_index = self.input_variables.index("theta")
        elif "buoyancy" in self.input_variables:
            self.theta_index = self.input_variables.index("buoyancy")
        else:
            self.theta_index = None

        self.metadata = [
            json.loads(item.decode("utf-8") if isinstance(item, bytes) else item)
            for item in self.h5[f"{split}/metadata_json"][:]
        ]

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        raw_x = self.x_data[idx]
        channels = []
        variable_mask = []
        for channel_index in self.channel_indices:
            if channel_index is None:
                channels.append(torch.zeros(raw_x.shape[-2:], dtype=torch.float32))
                variable_mask.append(False)
            else:
                channel = torch.tensor(raw_x[channel_index], dtype=torch.float32)
                has_finite_values = bool(torch.isfinite(channel).any())
                channel = torch.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
                channels.append(channel)
                variable_mask.append(has_finite_values)

        x = torch.stack(channels, dim=0)
        variable_mask = torch.tensor(variable_mask, dtype=torch.bool)
        meta = self.metadata[idx]

        _, _, condition = parse_case_parameters(meta["case_id"])

        #profile_label_key = (
        #    meta["case_id"],
        #    meta["plane"],
        #    int(meta["axis_index"]),
        #)
        case_label_key = meta["case_id"]

        if case_label_key in self.labels:
            #self.labels = {"KH:Ri012_a05_Re1000": 0.756281979832106}
            ratio = torch.tensor([self.labels[case_label_key]], dtype=torch.float32)
            has_ratio = torch.tensor(True)
        else:
            ratio = torch.tensor([0.0], dtype=torch.float32)
            has_ratio = torch.tensor(False)

        if self.theta_index is None:
            theta = torch.zeros((1, x.shape[-2], x.shape[-1]), dtype=torch.float32)
        else:
            theta = x[self.theta_index:self.theta_index + 1]

        return {
            "x": x,
            "theta": theta,
            "condition": condition,
            "ratio": ratio,
            "has_ratio": has_ratio,
            "variable_mask": variable_mask,
            "metadata": meta,
        }

    def close(self):
        self.h5.close()
