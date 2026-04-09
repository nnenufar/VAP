import ast
import json
import numpy as np
import pandas as pd


def build_session_lookup(agg_csv_path):
	usecols = [
		"vendor_id",
		"session_id",
		"interaction_id",
		"relationship_detail",
		"participant_ids",
		"participant_1_extraversion_raw",
		"participant_1_agreeableness_raw",
		"participant_1_conscientiousness_raw",
		"participant_1_neuroticism_raw",
		"participant_1_openness_raw",
		"participant_2_extraversion_raw",
		"participant_2_agreeableness_raw",
		"participant_2_conscientiousness_raw",
		"participant_2_neuroticism_raw",
		"participant_2_openness_raw",
	]
	df = pd.read_csv(
		agg_csv_path,
		usecols=usecols,
		dtype={"vendor_id": str, "session_id": str, "interaction_id": str},
	)

	df["full_session_id"] = (
		"V"
		+ df["vendor_id"].str.zfill(2)
		+ "_S"
		+ df["session_id"].str.zfill(4)
		+ "_I"
		+ df["interaction_id"].str.zfill(8)
	)

	df = df.drop_duplicates(subset="full_session_id")
	df = df.set_index("full_session_id")

	return df


def get_session_info(session_lookup_df, session_id):
	row = session_lookup_df.loc[session_id]
	personalities = [
		row["participant_1_extraversion_raw"],
		row["participant_1_agreeableness_raw"],
		row["participant_1_conscientiousness_raw"],
		row["participant_1_neuroticism_raw"],
		row["participant_1_openness_raw"],
		row["participant_2_extraversion_raw"],
		row["participant_2_agreeableness_raw"],
		row["participant_2_conscientiousness_raw"],
		row["participant_2_neuroticism_raw"],
		row["participant_2_openness_raw"],
	]
	return {
		"relationship": row["relationship_detail"],
		"participant_ids": row["participant_ids"],
		"personalities": personalities,
	}


def calculate_personality_statistics(slidingDset_csv_path, output_json_path = "data/personality_stats.json"):
	"""
	Calculates the mean and standard deviation for the 5 personality traits from both participants.
	Saves the result to a specified JSON file.
	"""
	df = pd.read_csv(slidingDset_csv_path)

	def parse_personalities(val):
		if isinstance(val, str):
			val = ast.literal_eval(val)
		return [float(x) for x in val]

	personalities_parsed = df["personalities"].apply(parse_personalities)

	traits = {
		"extraversion": [],
		"agreeableness": [],
		"conscientiousness": [],
		"neuroticism": [],
		"openness": []
	}

	for row in personalities_parsed:
		if len(row) == 10 and not np.isnan(row[0]):
			traits["extraversion"].extend([row[0], row[5]])
			traits["agreeableness"].extend([row[1], row[6]])
			traits["conscientiousness"].extend([row[2], row[7]])
			traits["neuroticism"].extend([row[3], row[8]])
			traits["openness"].extend([row[4], row[9]])

	stats = {}
	for trait_name, values in traits.items():
		stats[trait_name] = {
			"mean": float(np.nanmean(values)),
			"std": float(np.nanstd(values))
		}

	with open(output_json_path, 'w') as f:
		json.dump(stats, f, indent=4)

	return stats

