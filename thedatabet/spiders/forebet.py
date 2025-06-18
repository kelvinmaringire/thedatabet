import scrapy

import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from urllib.error import URLError

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import unicodedata


class ForebetSpider(scrapy.Spider):
    name = "forebet"

    def start_requests(self):
        # Dummy URL to trigger Scrapy
        yield scrapy.Request(url='https://example.com', callback=self.parse)

    def parse(self, response):

        print("⚡ Syncing SQLite with latest Forebet predictions")

        # 🗃️ Step 1: Connect to SQLite and create table
        conn = sqlite3.connect("forebet.db")
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_data (
            date TEXT PRIMARY KEY,
            data TEXT
        )
        """)
        conn.commit()

        # 📅 Step 2: Determine the next date to fetch from DB
        cursor.execute("SELECT date FROM match_data ORDER BY date DESC LIMIT 1")
        last_record = cursor.fetchone()
        if last_record:
            last_date = datetime.strptime(last_record[0], '%Y-%m-%d') + timedelta(days=1)
        else:
            # Default start if DB is empty
            last_date = datetime.strptime('2024-01-01', '%Y-%m-%d')

        # Set end date to yesterday
        end_date = datetime.now() - timedelta(days=1)

        # 🌐 Step 3: Prepare headers for request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        }

        # 🔁 Step 4: Loop through dates and fetch new data
        date_cursor = last_date
        new_days = 0

        while date_cursor <= end_date:
            date_str = date_cursor.strftime('%Y-%m-%d')
            url = f'https://www.forebet.com/scripts/getrs.php?ln=en&tp=1x2&in={date_str}&ord=0&tz=+120'

            req = Request(url, headers=headers)

            try:
                with urlopen(req) as response:
                    if response.status == 200:
                        response_data = json.loads(response.read().decode('utf-8'))

                        # ✅ Save to SQLite
                        cursor.execute(
                            "INSERT OR REPLACE INTO match_data (date, data) VALUES (?, ?)",
                            (date_str, json.dumps(response_data))
                        )
                        conn.commit()

                        new_days += 1
                        print(f"✅ Data saved for {date_str}")
                    else:
                        print(f"❌ Failed for {date_str} - Status code: {response.status}")
            except URLError as e:
                print(f"⚠️ Error retrieving data for {date_str}: {e.reason}")

            date_cursor += timedelta(days=1)

        # 📊 Step 5: Load into Pandas and summarize
        df = pd.read_sql_query("SELECT * FROM match_data", conn)

        # Convert JSON string back into lists/dicts
        df['data'] = df['data'].apply(json.loads)

        print("\n📦 Database Summary")
        print(f"🗓️ Total days stored: {len(df)}")
        print(f"🆕 New days added this run: {new_days}")
        print("📅 Date range:", df['date'].min(), "➡", df['date'].max())

        # 🔚 Clean up
        cursor.close()
        conn.close()

        print("🔄 Converting raw data to DataFrame...")

        games_array = []

        for row in df.itertuples(index=False):
            date = row.date
            data_for_date = row.data

            # The structure is assumed to be: [games_list]
            if isinstance(data_for_date, list) and len(data_for_date) > 0:
                games_list = data_for_date[0]  # same as data_for_date[0] in your JSON code
                games_array.extend(games_list)

        games_df = pd.DataFrame(
            games_array,
            columns=["DATE_BAH", "league_id", "Pred_1", "Pred_X", "Pred_2", "host_id", "guest_id", "HOST_NAME",
                     "GUEST_NAME", "Host_SC", "Guest_SC", "host_sc_pr", "guest_sc_pr", "goalsavg", "fctr", "Round",
                     "isCup", "is_nationalteam_cup", "is_international_club_cup"]
        )

        games_df.dropna(inplace=True)

        type_conversions = {
            'DATE_BAH': 'datetime64[ns]',
            'league_id': 'int16',
            'Pred_1': 'int8',
            'Pred_X': 'int8',
            'Pred_2': 'int8',
            'host_id': 'int16',
            'guest_id': 'int16',
            'Host_SC': 'int8',
            'Guest_SC': 'int8',
            'host_sc_pr': 'int8',
            'guest_sc_pr': 'int8',
            'goalsavg': 'float32',
            'fctr': 'int8',
            'Round': 'int8',
            'isCup': 'int8',
            'is_nationalteam_cup': 'int8',
            'is_international_club_cup': 'int8'
        }

        games_df = games_df.astype(type_conversions)


        print("⚽ Selecting relevant leagues...")

        """
        with open('leagues.json', 'r') as file:
            country_leagues = json.load(file)

        # First, extract all forebet league IDs from country_leagues
        forebet_league_ids = []
        for country in country_leagues.values():
            for league in country.values():
                forebet_league_ids.append(league['forebet'])

        # Now filter the DataFrame
        filtered_games_df = games_df[games_df['league_id'].isin(forebet_league_ids)].copy()

        # If you want to reset the index after filtering
        filtered_games_df.reset_index(drop=True, inplace=True)
        """
        filtered_games_df = games_df

        print("🔧 Building predictive features...")

        # Sort by date to ensure rolling calculations are correct chronologically
        filtered_games_df = filtered_games_df.sort_values(by='DATE_BAH').reset_index(drop=True)

        # --- Intermediate calculations (temporary variables) ---

        # 1. Calculate match result: 1=home win, 2=draw, 3=away win
        result = np.select(
            [filtered_games_df['Host_SC'] > filtered_games_df['Guest_SC'],
             filtered_games_df['Host_SC'] == filtered_games_df['Guest_SC'],
             filtered_games_df['Host_SC'] < filtered_games_df['Guest_SC']],
            [1, 2, 3]
        )

        # 2. Map match results to points (win=3, draw=1, loss=0)
        home_points = np.select(
            [result == 1, result == 2, result == 3],
            [3, 1, 0]
        )

        guest_points = np.select(
            [result == 3, result == 2, result == 1],
            [3, 1, 0]
        )

        # 3. Calculate rolling averages of points (form) over last 5 matches per team
        home_points_rolling = (
            filtered_games_df.assign(home_points=home_points)
            .groupby('host_id')['home_points']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        guest_points_rolling = (
            filtered_games_df.assign(guest_points=guest_points)
            .groupby('guest_id')['guest_points']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # 4. Calculate rolling averages of goals scored and conceded over last 5 matches per team

        # Host team performance (goals scored at home)
        host_perfom = (
            filtered_games_df.groupby('host_id')['Host_SC']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Host team conceded goals at home (goals by guest)
        host_concede = (
            filtered_games_df.groupby('host_id')['Guest_SC']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Guest team performance (goals scored away)
        guest_perfom = (
            filtered_games_df.groupby('guest_id')['Guest_SC']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Guest team conceded goals away (goals by host)
        guest_concede = (
            filtered_games_df.groupby('guest_id')['Host_SC']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # 5 Calculate rolling average of goal difference over last 5 matches per team

        # Host goal difference = goals scored at home - goals conceded at home
        host_goal_diff = (
            filtered_games_df.assign(goal_diff=filtered_games_df['Host_SC'] - filtered_games_df['Guest_SC'])
            .groupby('host_id')['goal_diff']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Guest goal difference = goals scored away - goals conceded away
        guest_goal_diff = (
            filtered_games_df.assign(goal_diff=filtered_games_df['Guest_SC'] - filtered_games_df['Host_SC'])
            .groupby('guest_id')['goal_diff']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # --- Add only final features to the DataFrame ---
        filtered_games_df = filtered_games_df.assign(
            home_result_rolling=home_points_rolling.fillna(0),
            guest_result_rolling=guest_points_rolling.fillna(0),
            Host_Perfom=host_perfom.fillna(0),
            Host_Concede=host_concede.fillna(0),
            Guest_Perfom=guest_perfom.fillna(0),
            Guest_Concede=guest_concede.fillna(0),
            Host_Goal_Diff=host_goal_diff.fillna(0),
            Guest_Goal_Diff=guest_goal_diff.fillna(0)
        )

        print("🤖 Training prediction models...")

        df = filtered_games_df

        # -----------------------
        # Define classification labels
        # -----------------------
        df['home_win'] = (df['Host_SC'] > df['Guest_SC']).astype(int)
        df['away_win'] = (df['Guest_SC'] > df['Host_SC']).astype(int)
        df['btts_yes'] = ((df['Host_SC'] > 0) & (df['Guest_SC'] > 0)).astype(int)
        df['btts_no'] = ((df['Host_SC'] == 0) | (df['Guest_SC'] == 0)).astype(int)
        df['over_2_5'] = ((df['Host_SC'] + df['Guest_SC']) > 2.5).astype(int)
        df['under_2_5'] = ((df['Host_SC'] + df['Guest_SC']) < 2.5).astype(int)
        df['home_over_1_5'] = (df['Host_SC'] > 1.5).astype(int)
        df['away_over_1_5'] = (df['Guest_SC'] > 1.5).astype(int)

        # -----------------------
        # Prepare features and targets
        # -----------------------
        features = df.drop(columns=[
            "Host_SC", "Guest_SC", "HOST_NAME", "GUEST_NAME", "DATE_BAH",
            "home_win", "away_win", "btts_yes", "btts_no", "over_2_5", "under_2_5", "home_over_1_5", "away_over_1_5"
        ])

        market_targets = [
            'home_win', 'away_win',
            'btts_yes', 'btts_no',
            'over_2_5', 'under_2_5',
            'home_over_1_5', 'away_over_1_5'
        ]

        # -----------------------
        # Train classifiers
        # -----------------------
        models = {}
        classification_metrics = {}

        for market in market_targets:
            X_train, X_test, y_train, y_test = train_test_split(
                features, df[market], test_size=0.3, random_state=42
            )

            model = lgb.LGBMClassifier(
                objective="binary",
                learning_rate=0.05,
                num_leaves=31,  # Keep this
                max_depth=5,  # Try limiting depth
                n_estimators=200,
                min_child_samples=30,  # Increased to prevent overfitting
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                class_weight='balanced',
                verbose=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store the model
            models[market] = model

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            classification_metrics[market] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            }

        # -----------------------
        # Show results
        # -----------------------
        print("📊 Classification metrics:")
        for market, metrics in classification_metrics.items():
            print(f"\n{market}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        # -----------------------
        # Predict on full dataset
        # -----------------------
        for market, model in models.items():
            df[f"Predicted_{market}"] = model.predict(features)
            df[f"{market}_prob"] = model.predict_proba(features)[:, 1]


        print("🌐 Fetching tomorrow's predictions...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        }

        # Get today's date
        today = datetime.now()

        # Calculate the date for the next day
        next_day = today + timedelta(days=1)

        # Format the date in the required format (YYYY-MM-DD)
        next_day_str = next_day.strftime('%Y-%m-%d')

        # Construct the URL with the next day's date
        url = f'https://www.forebet.com/scripts/getrs.php?ln=en&tp=1x2&in={next_day_str}&ord=0&tz=+120'

        try:
            # Create request object
            req = Request(url, headers=headers)

            # Make the request
            with urlopen(req) as response:
                # Check if request was successful (status code 200)
                if response.getcode() == 200:
                    # Read and decode the response
                    data = response.read().decode('utf-8')
                    # Parse JSON if needed
                    json_data = json.loads(data)
                    print("Successfully retrieved data for", next_day_str)
                else:
                    print(f"Request failed with status code: {response.getcode()}")

        except URLError as e:
            print(f"Error making request: {e.reason}")


        print("🔄 Normalizing and filtering new games...")

        new_games = json_data[0]
        new_leagues = json_data[1]

        new_games_df = pd.DataFrame(new_games,
                                    columns=["DATE_BAH", "league_id", "Pred_1", "Pred_X", "Pred_2", "host_id",
                                             "guest_id", "HOST_NAME", "GUEST_NAME", "host_sc_pr", "guest_sc_pr",
                                             "goalsavg", "fctr", "Round", "isCup", "is_nationalteam_cup",
                                             "is_international_club_cup"])

        def remove_accents(text):
            if pd.isna(text):
                return text
            # Normalize the text to NFKD (decomposed) form and encode only ASCII characters
            return ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if not unicodedata.combining(c)
            )

        # Apply to both HOST_NAME and GUEST_NAME columns
        new_games_df['HOST_NAME'] = new_games_df['HOST_NAME'].apply(remove_accents)
        new_games_df['GUEST_NAME'] = new_games_df['GUEST_NAME'].apply(remove_accents)

        # Check for null values in each column
        new_games_df.dropna(inplace=True)

        # Add league and Country in the dataframe
        new_leagues_df = pd.DataFrame(new_leagues)
        column_name = new_leagues_df.columns[0]
        first_row_value = new_leagues_df.iloc[1][column_name]
        new_games_df["country"] = new_games_df["league_id"].map(new_leagues_df.iloc[0])
        new_games_df["league"] = new_games_df["league_id"].map(new_leagues_df.iloc[1])

        type_conversions = {
            'DATE_BAH': 'datetime64[ns]',
            'league_id': 'int16',
            'Pred_1': 'int8',
            'Pred_X': 'int8',
            'Pred_2': 'int8',
            'host_id': 'int16',
            'guest_id': 'int16',
            'host_sc_pr': 'int8',
            'guest_sc_pr': 'int8',
            'goalsavg': 'float32',
            'fctr': 'int8',
            'Round': 'int8',
            'isCup': 'int8',
            'is_nationalteam_cup': 'int8',
            'is_international_club_cup': 'int8'
        }

        new_games_df = new_games_df.astype(type_conversions)
        # Now filter the DataFrame
        # filtered_new_games_df = new_games_df[new_games_df['league_id'].isin(forebet_league_ids)].copy()

        # If you want to reset the index after filtering
        # filtered_new_games_df.reset_index(drop=True, inplace=True)
        filtered_new_games_df = new_games_df

        print("🔄 Joining historical performance data...")

        for index, row in filtered_new_games_df.iterrows():
            host_id = row['host_id']
            guest_id = row['guest_id']

            # Filter rows in df for the current host_id and guest_id
            host_rows = df[df['host_id'] == host_id]
            guest_rows = df[df['guest_id'] == guest_id]

            # Check if there are matching rows and if not, assign NaN or any default value
            host_perfom = host_rows['Host_Perfom'].iloc[-1] if not host_rows.empty else None
            host_concede = host_rows['Host_Concede'].iloc[-1] if not host_rows.empty else None
            guest_perfom = guest_rows['Guest_Perfom'].iloc[-1] if not guest_rows.empty else None
            guest_concede = guest_rows['Guest_Concede'].iloc[-1] if not guest_rows.empty else None
            home_result_rolling = guest_rows['home_result_rolling'].iloc[-1] if not guest_rows.empty else None
            guest_result_rolling = guest_rows['guest_result_rolling'].iloc[-1] if not guest_rows.empty else None
            host_goal_diff = host_rows['Host_Goal_Diff'].iloc[-1] if not host_rows.empty else None
            guest_goal_diff = guest_rows['Guest_Goal_Diff'].iloc[-1] if not guest_rows.empty else None

            # Assign the performance data to the corresponding rows in filtered_new_games_df
            filtered_new_games_df.at[index, 'Host_Perfom'] = host_perfom
            filtered_new_games_df.at[index, 'Host_Concede'] = host_concede
            filtered_new_games_df.at[index, 'Guest_Perfom'] = guest_perfom
            filtered_new_games_df.at[index, 'Guest_Concede'] = guest_concede
            filtered_new_games_df.at[index, 'home_result_rolling'] = home_result_rolling
            filtered_new_games_df.at[index, 'guest_result_rolling'] = guest_result_rolling
            filtered_new_games_df.at[index, 'Host_Goal_Diff'] = host_goal_diff
            filtered_new_games_df.at[index, 'Guest_Goal_Diff'] = guest_goal_diff

        # Drop rows with missing values in the specified columns
        filtered_new_games_df.dropna(subset=[
            'Host_Perfom', 'Host_Concede', 'Guest_Perfom', 'Guest_Concede',
            'home_result_rolling', 'guest_result_rolling',
            'Host_Goal_Diff', 'Guest_Goal_Diff'
        ], inplace=True)


        print("🤖 Applying trained models...")

        # Ensure new_games_df has the same feature columns as used during training
        # Drop any columns not used during model training
        new_features = filtered_new_games_df.drop(columns=[
            "HOST_NAME", "GUEST_NAME", "DATE_BAH", "country", "league"
        ])

        # Make sure column order matches the training features
        new_features = new_features[features.columns]

        # Predict using trained models and append predictions
        for market, model in models.items():
            filtered_new_games_df[f"Predicted_{market}"] = model.predict(new_features)
            filtered_new_games_df[f"{market}_prob"] = model.predict_proba(new_features)[:, 1]

        # To JSON format
        new_json_data = filtered_new_games_df.to_json(orient='records')

        print("✨ All predictions processed and saved")

        # Yield results
        yield {
            'data': json.loads(new_json_data)
        }
