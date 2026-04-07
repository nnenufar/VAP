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
