from datetime import datetime, timedelta
from .supabase_config import supabase  # import the client


class SupabasePipeline:

    def __init__(self):
        # Set tomorrow's date as "dd-mm-yyyy"
        self.tomorrow_str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        self.items = []

    def process_item(self, item, spider):
        self.items.append(item)
        return item

    def close_spider(self, spider):
        if self.items:
            try:
                supabase.table("betway_data").upsert({
                    "date_key": self.tomorrow_str,
                    "data": self.items
                }, on_conflict=["date_key"]).execute()
            except Exception as e:
                print("Failed to upsert into Supabase:", e)
