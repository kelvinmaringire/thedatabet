import os
import json
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime, timedelta
import scrapy
from scrapy.selector import Selector


class BetwaySpider(scrapy.Spider):
    name = "betway"
    allowed_domains = ["betway.co.za"]

    def start_requests(self):

        # Get the current directory of the script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the correct path to the JSON file
        json_file_path = os.path.join(current_dir, "leagues.json")

        try:
            # Read and load the JSON file
            with open(json_file_path, "r", encoding="utf-8") as file:
                country_leagues = json.load(file)

        except FileNotFoundError:
            print(f"Error: The file {json_file_path} was not found.")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON file. {e}")


            # Iterate through countries and leagues
        for country, leagues in country_leagues.items():
            for league_name, league_data in leagues.items():
                betway_id = league_data['betway']

                # Construct the URL
                league_url = (
                    f"https://www.betway.co.za/Event/FilterEventsGet?"
                    f"sportConfigId=00000000-0000-0000-da7a-000000550001&"
                    f"feedDataTypeId=00000000-0000-0000-da7a-000000580001&"
                    f"leagueIds={betway_id}&"
                    f"PredeterminedTime=None"
                )

                yield scrapy.Request(
                    url=league_url,
                    callback=self.parse
                )

    def parse(self, response):

        selector = Selector(response)
        # Extract the desired data using CSS or XPath selectors
        relative_game_urls = selector.css(
            'div#fixturesToReplace div.eventRow div#eventDetails_0 > div.inplayStatusDetails.PaddingScreen > a::attr(href)'
        ).getall()

        today = datetime.today()
        tomorrow = today + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y%m%d")
        filtered_urls = [url for url in relative_game_urls if f"datefilter={tomorrow_str}" in url]  # unused
        base_url = 'https://www.betway.co.za/'

        for relative_game_url in filtered_urls:
            absolute_url = urljoin(base_url, relative_game_url)
            parsed_url = urlparse(absolute_url)
            path_components = parsed_url.path.split('/')
            country_code = path_components[3]
            league = path_components[4]
            league_without_underscores = league.replace("_", " ")
            capitalized_league = league_without_underscores.capitalize()
            query_params = parse_qs(parsed_url.query)
            date_time = query_params.get('datefilter', [''])[0]
            event_id = query_params.get('eventId', [''])[0]
            date = date_time[:8]
            time = date_time[8:]

            original_url = 'https://www.betway.co.za/Bet/EventMultiMarket?' \
                           'eventId=026e4607-0000-0000-0000-000000000000&' \
                           'FeedDataTypeId=00000000-0000-0000-0000-000000000000&' \
                           'isPopular=false&pageNum=1&isFullView=false&loadAll=true'
            new_url = original_url.replace('eventId=026e4607-0000-0000-0000-000000000000', f'eventId={event_id}')

            #print(new_url)

            yield scrapy.Request(new_url, callback=self.parse_event, meta={
                'item': {
                    'Country Code': country_code,
                    'League': capitalized_league,
                    'Date': date,
                    'Time': time,
                    'Event ID': event_id
                }
            })

    def parse_event(self, response):
        # Extract the desired data from the response
        country_code = response.meta['item']['Country Code']
        league = response.meta['item']['League']
        date = response.meta['item']['Date']
        time = response.meta['item']['Time']

        home_draw_away_elements = response.css('[data-translate-market="Match Result (1X2)"]').getall()

        home_element = home_draw_away_elements[0]
        away_element = home_draw_away_elements[2]

        home_selector = Selector(text=home_element)
        away_selector = Selector(text=away_element)

        # Extract the text of the data-translate-key attribute
        host_name = home_selector.css('span::attr(data-translate-key)').get()
        guest_name = away_selector.css('span::attr(data-translate-key)').get()

        # Home odds
        home_odds_target_element = response.xpath(
            '//span[@data-translate-market="Match Result (1X2)" and @data-translate-key=$team]', team=host_name)
        home_odds_parent_xpath = home_odds_target_element.xpath('parent::node()').xpath('parent::node()')
        home_odds_element_with_new_line = home_odds_parent_xpath.css('div.outcome-pricedecimal::text').get()
        try:
            home_odds = float(home_odds_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            home_odds = None


        # Away odds
        away_odds_target_element = response.xpath(
            '//span[@data-translate-market="Match Result (1X2)" and @data-translate-key=$team]', team=guest_name)
        away_odds_parent_xpath = away_odds_target_element.xpath('parent::node()').xpath('parent::node()')
        away_odds_element_with_new_line = away_odds_parent_xpath.css('div.outcome-pricedecimal::text').get()
        try:
            away_odds = float(away_odds_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            away_odds = None

        double_chances = response.css("[data-translate-market='Double Chance']" "[data-translate-key='Draw']")
        if double_chances:
            try:
                home_draw_target_element = double_chances[0]
                home_draw_parent_locator = home_draw_target_element.xpath('parent::node()').xpath('parent::node()')
                home_draw_element_with_new_line = home_draw_parent_locator.css('div.outcome-pricedecimal::text').get()
                home_draw = float(home_draw_element_with_new_line.replace('\n', '').replace('\r', '').strip())
            except AttributeError:
                home_draw = None

            try:
                away_draw_target_element = double_chances[1]
                away_draw_parent_locator = away_draw_target_element.xpath('parent::node()').xpath('parent::node()')
                away_draw_element_with_new_line = away_draw_parent_locator.css('div.outcome-pricedecimal::text').get()
                away_draw = float(away_draw_element_with_new_line.replace('\n', '').replace('\r', '').strip())
            except AttributeError:
                away_draw = None

        else:
            home_draw = None
            away_draw = None


        # Over 1.5
        over15_target_element = response.css("[data-translate-market='Overs/Unders (Total 1.5)']" "[data-translate-key='Over 1.5']")
        over15_parent_xpath = over15_target_element.xpath('parent::node()').xpath('parent::node()')
        over15_element_with_new_line = over15_parent_xpath.css('div.outcome-pricedecimal::text').get()
        try:
            over15 = float(over15_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            over15 = None

        # Under 3.5
        under35_target_element = response.css(
            "[data-translate-market='Overs/Unders (Total 3.5)']" "[data-translate-key='Under 3.5']")
        under35_parent_xpath = under35_target_element.xpath('parent::node()').xpath('parent::node()')
        under35_element_with_new_line = under35_parent_xpath.css('div.outcome-pricedecimal::text').get()
        try:
            under35 = float(under35_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            under35 = None

        # Home Total Over 0.5 odds
        home_total_over05_odds_target_element = response.css(
            f'[data-translate-market="{host_name} Total (Total 0.5)"]' '[data-translate-key="Over 0.5"]')
        home_total_over05_odds_parent_xpath = home_total_over05_odds_target_element.xpath('parent::node()').xpath(
            'parent::node()')
        home_total_over05_odds_element_with_new_line = home_total_over05_odds_parent_xpath.css(
            'div.outcome-pricedecimal::text').get()
        try:
            home_total_over05_odds = float(home_total_over05_odds_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            home_total_over05_odds = None

        # Away Total Over 0.5 odds
        away_total_over05_odds_target_element = response.css(
             f'[data-translate-market="{guest_name} Total (Total 0.5)"]' '[data-translate-key="Over 0.5"]')
        away_total_over05_odds_parent_xpath = away_total_over05_odds_target_element.xpath('parent::node()').xpath(
            'parent::node()')
        away_total_over05_odds_element_with_new_line = away_total_over05_odds_parent_xpath.css(
            'div.outcome-pricedecimal::text').get()
        try:
            away_total_over05_odds = float(away_total_over05_odds_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            away_total_over05_odds = None


        # BTTS Yes
        btts_yes_target_element = response.css(
            '[data-translate-key="Yes"]' '[data-translate-market="Both Teams To Score"]')
        btts_yes_parent_xpath = btts_yes_target_element.xpath('parent::node()').xpath('parent::node()')
        btts_yes_element_with_new_line = btts_yes_parent_xpath.css('div.outcome-pricedecimal::text').get()
        try:
            btts_yes = float(btts_yes_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            btts_yes = None

        # BTTS No
        btts_no_target_element = response.css(
            '[data-translate-key="No"]' '[data-translate-market="Both Teams To Score"]')
        btts_no_parent_xpath = btts_no_target_element.xpath('parent::node()').xpath('parent::node()')
        btts_no_element_with_new_line = btts_no_parent_xpath.css('div.outcome-pricedecimal::text').get()
        try:
            btts_no = float(btts_no_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        except AttributeError:
            btts_no = None

        # Draw no bet
        draw_no_bet_odds_target_elements = response.css('[data-translate-market="Draw No Bet"]')

        # Home Draw no bet odds
        if len(draw_no_bet_odds_target_elements) >= 1:
            home_draw_no_bet_odds_target_element = draw_no_bet_odds_target_elements[0]
            home_draw_no_bet_odds_parent_xpath = home_draw_no_bet_odds_target_element.xpath('parent::node()').xpath(
                'parent::node()')
            home_draw_no_bet_odds_element_with_new_line = home_draw_no_bet_odds_parent_xpath.css(
                'div.outcome-pricedecimal::text').get()
            home_draw_no_bet_odds = float(home_draw_no_bet_odds_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        else:
            home_draw_no_bet_odds = None

        # Away Draw no bet odds
        if len(draw_no_bet_odds_target_elements) >= 2:
            away_draw_no_bet_odds_target_element = draw_no_bet_odds_target_elements[1]
            away_draw_no_bet_odds_parent_xpath = away_draw_no_bet_odds_target_element.xpath('parent::node()').xpath(
                'parent::node()')
            away_draw_no_bet_odds_element_with_new_line = away_draw_no_bet_odds_parent_xpath.css(
                'div.outcome-pricedecimal::text').get()
            away_draw_no_bet_odds = float(away_draw_no_bet_odds_element_with_new_line.replace('\n', '').replace('\r', '').strip())
        else:
            away_draw_no_bet_odds = None


        yield {
            'country_code': country_code,
            'betway_league': league,
            'date': date,
            'time': time,
            'host_name': host_name,
            'guest_name': guest_name,
            'home_odds': home_odds,
            'home_draw': home_draw,
            'away_draw': away_draw,
            'away_odds': away_odds,
            'over_15': over15,
            'under_35': under35,
            'home05': home_total_over05_odds,
            'away05': away_total_over05_odds,
            'btts_y': btts_yes,
            'btts_n': btts_no,
            'home_draw_no_bet': home_draw_no_bet_odds,
            'away_draw_no_bet': away_draw_no_bet_odds,

        }

# scrapy runspider betway.py -O output.json
