import scrapy
import json

import pandas as pd
import requests
from datetime import datetime, timedelta

from rapidfuzz import process, fuzz

from thedatabet.supabase_config import SUPABASE_URL, headers


class FootballtipsSpider(scrapy.Spider):
    name = "footballtips"

    def start_requests(self):
        # Dummy URL to trigger Scrapy
        yield scrapy.Request(url='https://example.com', callback=self.parse)

    def parse(self, response):

        print("✅ Loaded Betway odds and Forebet tips for")

        today = datetime.now()
        next_day = today + timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')

        # Query with a filter on the date_key column
        params = {
            "date_key": f"eq.{next_day_str}"
        }

        # Make all the requests
        betway_data = requests.get(
            f"{SUPABASE_URL}/rest/v1/betway_data",
            headers=headers,
            params=params
        )

        forebet_data = requests.get(
            f"{SUPABASE_URL}/rest/v1/forebet_data",
            headers=headers,
            params=params
        )

        if betway_data.status_code == 200:
            betway_odds = betway_data.json()
        else:
            print(f"Error in betway_data: {betway_data.status_code}, {betway_data.text}")

        if forebet_data.status_code == 200:
            forebet_tips = forebet_data.json()
        else:
            print(f"Error in forebet_data: {forebet_data.status_code}, {forebet_data.text}")

        betway_odds_df = pd.DataFrame(betway_odds[0]['data'])
        forebet_tips_df = pd.DataFrame(forebet_tips[0]['data'][0]['data'])

        print("🧹 Cleaned team names in forebet_tips_df and betway_odds_df")

        # List of suffixes and prefixes to remove (uppercased for matching)
        suffixes_and_prefixes = [
            ' FC', ' SC', ' CD', ' W', ' CF', ' AE', ' CA', ' FK', ' MC', ' CR',
            ' EC', ' PR', ' SP', ' SE', ' RS', ' AD', ' FE', ' SK', ' LFS', ' RJ', ' CS'
        ]

        # Normalize affixes (strip + upper)
        suffixes_and_prefixes = [affix.strip().upper() for affix in suffixes_and_prefixes]

        def clean_team_name(name):
            if pd.isna(name):
                return name
            cleaned_name = name.strip()

            while True:
                original_name = cleaned_name
                # Uppercase name for matching, keep original casing for return
                upper_name = cleaned_name.upper()

                for affix in suffixes_and_prefixes:
                    # Check and remove prefix
                    if upper_name.startswith(affix + ' '):
                        cleaned_name = cleaned_name[len(affix) + 1:].strip()
                        upper_name = cleaned_name.upper()
                    # Check and remove suffix
                    elif upper_name.endswith(' ' + affix):
                        cleaned_name = cleaned_name[: -len(affix) - 1].strip()
                        upper_name = cleaned_name.upper()

                if cleaned_name == original_name:
                    break
            return cleaned_name

        # Example usage
        # Apply to DataFrame columns
        forebet_tips_df['HOST_NAME'] = forebet_tips_df['HOST_NAME'].apply(clean_team_name)
        forebet_tips_df['GUEST_NAME'] = forebet_tips_df['GUEST_NAME'].apply(clean_team_name)
        betway_odds_df['host_name'] = betway_odds_df['host_name'].apply(clean_team_name)
        betway_odds_df['guest_name'] = betway_odds_df['guest_name'].apply(clean_team_name)

        print("⚡ Fuzzy-matched odds predictions (score ≥75)")

        # Assume forebet_tips_df and betway_odds_df are already defined DataFrames
        score_threshold = 75

        # Track indices of betway_odds_df that have already been matched
        used_betway_indices = set()
        matches = []

        for idx, forebet_row in forebet_tips_df.iterrows():
            host_name = forebet_row['HOST_NAME']
            guest_name = forebet_row['GUEST_NAME']

            best_match = None
            best_score = 0
            match_type = None  # 'host' or 'guest'
            matched_betway_idx = None

            # Prepare the pool of available (unmatched) betway rows
            available = betway_odds_df[~betway_odds_df.index.isin(used_betway_indices)]

            # 1. Try matching HOST_NAME to betway host_name
            if pd.notna(host_name) and not available.empty:
                choices = dict(zip(available.index, available['host_name']))
                match = process.extractOne(host_name, choices, scorer=fuzz.token_sort_ratio,
                                           score_cutoff=score_threshold)
                if match:
                    matched_value, score, betway_idx = match
                    best_match = matched_value
                    best_score = score
                    match_type = 'host'
                    matched_betway_idx = betway_idx

            # 2. If no host match found, try matching GUEST_NAME to betway guest_name
            if best_match is None and pd.notna(guest_name) and not available.empty:
                available = betway_odds_df[~betway_odds_df.index.isin(used_betway_indices)]
                choices = dict(zip(available.index, available['guest_name']))
                match = process.extractOne(guest_name, choices, scorer=fuzz.token_sort_ratio,
                                           score_cutoff=score_threshold)
                if match:
                    matched_value, score, betway_idx = match
                    best_match = matched_value
                    best_score = score
                    match_type = 'guest'
                    matched_betway_idx = betway_idx

            # 3. If a match was found, record it and exclude that betway row
            if best_match is not None:
                used_betway_indices.add(matched_betway_idx)
                matched_betway_row = betway_odds_df.loc[matched_betway_idx]

                # Convert rows to dicts
                forebet_data = forebet_row.to_dict()
                betway_data = matched_betway_row.to_dict()

                # Add match metadata
                match_info = {
                    'MATCH_TYPE': match_type,
                    'SCORE': best_score
                }

                # Combine all data into one dictionary
                combined_row = {**forebet_data, **match_info, **betway_data}
                matches.append(combined_row)

        # Create final DataFrame of matches
        final_matched_games_df = pd.DataFrame(matches)
        cols_to_round = [
            'goalsavg', 'btts_no_prob', 'away_win_prob', 'btts_yes_prob',
            'home_win_prob', 'over_2_5_prob', 'under_2_5_prob',
            'away_over_1_5_prob', 'home_over_1_5_prob', 'SCORE'
        ]

        # Round each column if it exists in the DataFrame
        for col in cols_to_round:
            if col in final_matched_games_df.columns:
                final_matched_games_df[col] = final_matched_games_df[col].round(2)

        print("📈 Profitable Betting Opportunities")

        # Define minimum odds threshold per market
        MIN_ODDS_THRESHOLD = {
            'Predicted_home_win': 1.70,
            'Predicted_away_win': 1.85,
            'Predicted_btts_yes': 1.60,
            'Predicted_btts_no': 1.70,
            'Predicted_over_2_5': 1.18,
            'Predicted_under_2_5': 1.30,
            'Predicted_home_over_1_5': 1.20,
            'Predicted_away_over_1_5': 1.20
        }

        # Mapping of each prediction to its (probability column, odds column)
        market_rules = {
            'Predicted_home_win': ('home_win_prob', 'home_odds'),
            'Predicted_away_win': ('away_win_prob', 'away_odds'),
            'Predicted_btts_yes': ('btts_yes_prob', 'btts_y'),
            'Predicted_btts_no': ('btts_no_prob', 'btts_n'),
            'Predicted_over_2_5': ('over_2_5_prob', 'over_15'),
            'Predicted_under_2_5': ('under_2_5_prob', 'under_35'),
            'Predicted_home_over_1_5': ('home_over_1_5_prob', 'home05'),
            'Predicted_away_over_1_5': ('away_over_1_5_prob', 'away05')
        }

        # Track changes for metrics
        ev_filtering_summary = {}

        for pred_col, (prob_col, odds_col) in market_rules.items():
            min_threshold = MIN_ODDS_THRESHOLD[pred_col]

            # Mask for predictions = 1 and all required data is present
            mask = (
                    (final_matched_games_df[pred_col] == 1) &
                    final_matched_games_df[prob_col].notna() &
                    final_matched_games_df[odds_col].notna() &
                    (final_matched_games_df[prob_col] > 0)
            )

            sub_df = final_matched_games_df.loc[mask, [prob_col, odds_col]]

            # Calculate Expected Value (EV)
            ev = (sub_df[prob_col] * (sub_df[odds_col] - 1)) - (1 - sub_df[prob_col])

            # Build a mask for rows to keep: odds above threshold AND EV > 0
            to_keep_mask = (sub_df[odds_col] >= min_threshold) & (ev > 0)

            kept_count = to_keep_mask.sum()
            dropped_count = len(to_keep_mask) - kept_count

            # Update predictions where EV <= 0 or odds < threshold
            final_matched_games_df.loc[to_keep_mask.index[~to_keep_mask], pred_col] = 0

            # Calculate average EV for kept predictions
            avg_ev = ev[to_keep_mask].mean() if kept_count > 0 else None

            # Track results
            ev_filtering_summary[pred_col] = {
                'kept': int(kept_count),
                'dropped': int(dropped_count),
                'average_ev': round(avg_ev, 3) if avg_ev is not None else "N/A"
            }

        # Print final metrics
        print("📊 Prediction filtering based on MIN_ODDS_THRESHOLD + Expected Value (EV > 0):")
        for col, counts in ev_filtering_summary.items():
            print(f"- {col}: {counts['kept']} kept, {counts['dropped']} dropped, avg EV: {counts['average_ev']}")

        json_data = final_matched_games_df.to_json(orient='records')

        print("✨ All predictions processed and saved")

        # Yield results
        yield {
            'data': json.loads(json_data)
        }
