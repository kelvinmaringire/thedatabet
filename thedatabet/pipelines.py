from datetime import datetime, timedelta
from .supabase_config import supabase  # import the client


class SupabasePipeline:

    def __init__(self):
        # Set tomorrow's date in "YYYY-MM-DD" format
        self.tomorrow_str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Separate buffers for each spider
        self.data_by_spider = {
            "betway": [],
            "sportsmole": []
        }

    def process_item(self, item, spider):
        # Add the item to the appropriate buffer
        if spider.name in self.data_by_spider:
            self.data_by_spider[spider.name].append(item)
        else:
            print(f"Unknown spider '{spider.name}' – item skipped")
        return item

    def close_spider(self, spider):
        spider_name = spider.name
        items = self.data_by_spider.get(spider_name)

        if items:
            table_name = f"{spider_name}_data"  # e.g., betway_data or sportsmole_data

            try:
                supabase.table(table_name).upsert({
                    "date_key": self.tomorrow_str,
                    "data": items
                }, on_conflict=["date_key"]).execute()
                print(f"Upserted {len(items)} items into '{table_name}' for {self.tomorrow_str}")
            except Exception as e:
                print(f"Failed to upsert into Supabase table '{table_name}':", e)
