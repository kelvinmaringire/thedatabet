import scrapy
import json

import pandas as pd
import requests
from datetime import datetime, timedelta

from rapidfuzz import process, fuzz
from openai import OpenAI

from thedatabet.supabase_config import SUPABASE_URL, headers


class PreviewsSpider(scrapy.Spider):
    name = "previews"

    def start_requests(self):
        # Dummy URL to trigger Scrapy
        yield scrapy.Request(url='https://example.com', callback=self.parse)

    def parse(self, response):

        print("✅ Loaded Betway odds and Sportsmole tips for")

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

        sportsmole_data = requests.get(
            f"{SUPABASE_URL}/rest/v1/sportsmole_data",
            headers=headers,
            params=params
        )

        if betway_data.status_code == 200:
            betway_odds = betway_data.json()
        else:
            print(f"Error in betway_data: {betway_data.status_code}, {betway_data.text}")

        if sportsmole_data.status_code == 200:
            sportsmole_tips = sportsmole_data.json()
        else:
            print(f"Error in forebet_data: {sportsmole_data.status_code}, {sportsmole_data.text}")

        betway_odds_df = pd.DataFrame(betway_odds[0]['data'])
        sportsmole_tips_df = pd.DataFrame(sportsmole_tips[0]['data'])

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
        sportsmole_tips_df['home_team'] = sportsmole_tips_df['home_team'].apply(clean_team_name)
        sportsmole_tips_df['away_team'] = sportsmole_tips_df['away_team'].apply(clean_team_name)
        betway_odds_df['host_name'] = betway_odds_df['host_name'].apply(clean_team_name)
        betway_odds_df['guest_name'] = betway_odds_df['guest_name'].apply(clean_team_name)

        print("⚡ Fuzzy-matched odds predictions (score ≥75)")

        # Assume forebet_tips_df and betway_odds_df are already defined DataFrames
        score_threshold = 75

        # Track indices of betway_odds_df that have already been matched
        used_betway_indices = set()
        matches = []

        for idx, sportsmole_row in sportsmole_tips_df.iterrows():
            host_name = sportsmole_row['home_team']
            guest_name = sportsmole_row['away_team']

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
                sportsmole_data = sportsmole_row.to_dict()
                betway_data = matched_betway_row.to_dict()

                # Add match metadata
                match_info = {
                    'MATCH_TYPE': match_type,
                    'SCORE': best_score
                }

                # Combine all data into one dictionary
                combined_row = {**sportsmole_data, **match_info, **betway_data}
                matches.append(combined_row)

        # Create final DataFrame of matches
        final_matched_games_df = pd.DataFrame(matches)

        print("📈 Get previews from Open AI chatGPT gpt-3.5-turbo")

        # Initialize the client with your API key
        client = OpenAI(
            api_key="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")

        previews = []

        for _, row in final_matched_games_df.iterrows():
            match_dict = row.to_dict()

            prompt = (
                "You are a professional football tipster providing sharp betting analysis for serious punters. "
                "Analyze the match data to deliver concise, value-focused insights with these strict requirements:\n\n"

                "1. **STRUCTURE** (wrap entire response in a single <div> with this exact CSS):\n"
                "<div style='font-family: \"Roboto\", sans-serif; color: #e0e0e0; line-height: 1.5; max-width: 650px; margin: 0 auto; background-color: #1d1d1d; padding: 16px; border-radius: 8px;'>\n"
                "   <h3 style='color: #ff9800; border-bottom: 1px solid #333; padding-bottom: 8px;'>{{Home}} vs {{Away}}</h3>\n"
                "   <div style='background: #252525; padding: 12px; border-radius: 4px; margin-bottom: 15px; border-left: 3px solid #1976d2;'>\n"
                "       <!-- Key match facts -->\n"
                "   </div>\n"
                "   <div style='margin-bottom: 15px; background: #252525; padding: 12px; border-radius: 4px;'>\n"
                "       <!-- Data insights -->\n"
                "   </div>\n"
                "   <div style='background: #2a2a2a; padding: 12px; border-radius: 4px; border-left: 3px solid #ff9800;'>\n"
                "       <!-- Betting recommendation -->\n"
                "   </div>\n"
                "</div>\n\n"

                "2. **CONTENT RULES**:\n"
                "- Lead with 2-3 MOST RELEVANT match facts (injuries, trends, H2H)\n"
                "- Highlight 3-5 actionable data points (e.g. 'BTTS landed in 7/10 home games')\n"
                "- Recommend ONLY value bets where probability > implied odds probability\n"
                "- ALWAYS include current odds in recommendations (e.g. 'Home Win @ 2.10')\n"
                "- Use confidence levels: (Low <55%) (Medium 55-70%) (High >70%)\n"
                "- Color code confidence: <span style='color: #c10015'>Low</span>, <span style='color: #f2c037'>Medium</span>, <span style='color: #21ba45'>High</span>\n\n"

                "3. **TONE**:\n"
                "- Professional but direct - no fluff or summaries\n"
                "- Bullet points for data/metrics\n"
                "- Odds comparison like: '45% prob vs 2.10 odds (implied 47.6%)'\n"
                "- Never repeat the input verbatim"
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"Match Info:\n{match_dict}"
                    }
                ],
                temperature=0.5  # Lower for more factual analysis
            )

            previews.append(response.choices[0].message.content.strip())

        print("✨ All predictions processed and saved")

        # Yield results
        yield {
            'data': previews
        }
