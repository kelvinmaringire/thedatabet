import scrapy
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from urllib.error import URLError


class ForebetSpider(scrapy.Spider):
    name = "forebet"

    def start_requests(self):
        # Dummy URL to trigger Scrapy
        yield scrapy.Request(url='https://httpbin.org/get', callback=self.parse)

    def parse(self, response):
        # 🗃️ Connect to SQLite
        conn = sqlite3.connect("forebet.db")
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_data (
            date TEXT PRIMARY KEY,
            data TEXT
        )
        """)
        conn.commit()

        # 📅 Determine the next date to fetch
        cursor.execute("SELECT date FROM match_data ORDER BY date DESC LIMIT 1")
        last_record = cursor.fetchone()
        if last_record:
            last_date = datetime.strptime(last_record[0], '%Y-%m-%d') + timedelta(days=1)
        else:
            last_date = datetime.strptime('2024-01-01', '%Y-%m-%d')

        end_date = datetime.now() - timedelta(days=1)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        }

        # 🔁 Loop through dates and fetch new data
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

                        # ✅ Save to DB
                        cursor.execute(
                            "INSERT OR REPLACE INTO match_data (date, data) VALUES (?, ?)",
                            (date_str, json.dumps(response_data))
                        )
                        conn.commit()

                        new_days += 1
                        self.logger.info(f"✅ Data saved for {date_str}")
                    else:
                        self.logger.warning(f"❌ Failed for {date_str} - Status: {response.status}")
            except URLError as e:
                self.logger.warning(f"⚠️ Error for {date_str}: {e.reason}")

            date_cursor += timedelta(days=1)

        # 📊 Load from DB into Pandas
        df = pd.read_sql_query("SELECT * FROM match_data", conn)
        df['data'] = df['data'].apply(json.loads)

        # 🔚 Clean up DB connection
        cursor.close()
        conn.close()

        # Flatten game data
        games_array = []
        for row in df.itertuples(index=False):
            data_for_date = row.data
            if isinstance(data_for_date, list) and len(data_for_date) > 0:
                games_list = data_for_date[0]
                games_array.extend(games_list)

        games_df = pd.DataFrame(
            games_array,
            columns=[
                "DATE_BAH", "league_id", "Pred_1", "Pred_X", "Pred_2", "host_id", "guest_id",
                "HOST_NAME", "GUEST_NAME", "Host_SC", "Guest_SC", "host_sc_pr", "guest_sc_pr",
                "goalsavg", "fctr", "Round", "isCup", "is_nationalteam_cup", "is_international_club_cup"
            ]
        )
        games_df.dropna(inplace=True)

        # Take preview
        first_5_rows = games_df.head(5)
        json_data = first_5_rows.to_json(orient='records')

        # Yield results
        yield {
            'summary': {
                'total_days_stored': len(df),
                'new_days_added': new_days,
                'date_range': [df['date'].min(), df['date'].max()]
            },
            'games_preview': json.loads(json_data)
        }
