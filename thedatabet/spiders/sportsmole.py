import re
import scrapy


class SportsmoleSpider(scrapy.Spider):
    name = "sportsmole"
    allowed_domains = ["www.sportsmole.co.uk"]
    start_urls = ["https://www.sportsmole.co.uk/football/preview/"]

    def parse(self, response):
        # Get only football preview links (excludes subscribe and basketball)
        absolute_urls = [response.urljoin(href) for href in response.css('div.previews a::attr(href)').getall()]

        football_urls = [
            response.urljoin(href)
            for href in absolute_urls
            if "/football/" in href
               and "subscribe" not in href
               and not href.startswith("/basketball/")
        ]

        for preview_url in football_urls:
            yield scrapy.Request(
                url=preview_url,
                callback=self.parse_event
            )

    def parse_event(self, response):

        match_time = response.css('span[itemprop="startDate"]::attr(datetime)').get()
        league_name = response.css('div.w320.mAuto div:first-child::text').get().split("|")[0].strip()
        data_analysis_url = response.urljoin(
            response.xpath(
                '//p[contains(text(), "For data analysis of the most likely results, scorelines and more for this match")]/a/@href'
            ).get()
        )
        home_team = response.css('div.gb_team.left::text').get().strip()
        away_team = response.css('div.gb_team.right::text').get().strip()


        # Extract text including text from child elements (like <a> tags)
        paragraphs = response.css('div#article_body p *::text, div#article_body p::text').getall()
        full_text = ' '.join([p.strip() for p in paragraphs if p.strip()])

        # Remove any remaining HTML tags (handles cases where tags might still exist)
        clean_text = re.sub(r'<[^>]+>', '', full_text)

        # Normalize whitespace
        clean_text = ' '.join(clean_text.split())

        prediction_text = response.css('h2.sm_center::text').get().strip()

        combined_paragraph = f"{clean_text} {prediction_text}"

        yield scrapy.Request(
            url=data_analysis_url,
            callback=self.parse_data_analysis,
            meta={
                "match_time": match_time,
                "home_team": home_team,
                "away_team": away_team,
                "league_name": league_name,
                "combined_paragraph": combined_paragraph
            }
        )

    def parse_data_analysis(self, response):
        match_time = response.meta["match_time"]
        home_team = response.meta["home_team"]
        away_team = response.meta["away_team"]
        league_name = response.meta["league_name"]
        combined_paragraph = response.meta["combined_paragraph"]

        def extract_team_form(team_selector):
            form_data = {}
            # Loop through each competition block
            for section in response.css(f"{team_selector} div.lastSixOut"):
                comp_name = section.css("div.last_six a::text").get()
                results = section.css("ul.fg_results li span.wRd::text").getall()
                form_data[comp_name] = results
            return form_data

        home_form = extract_team_form("div.hw.game.left.gray")
        away_form = extract_team_form("div.hw.game.gray")

        item = {}

        # Extract result probabilities
        item['home_win_prob'] = response.css('td.homBG.lc_block::text').re_first(r'([\d.]+%)')
        item['draw_prob'] = response.css('td.drwBG.lc_block::text').re_first(r'([\d.]+%)')
        item['away_win_prob'] = response.css('td.awaBG.lc_block::text').re_first(r'([\d.]+%)')

        # Extract both teams to score
        item['both_teams_to_score'] = response.xpath(
            '//td[contains(@class, "homBG") and contains(span/text(), "Both teams to score")]/span/text()'
        ).re_first(r'([\d.]+%)')

        # Extract over/under goals
        item['over_2.5'] = response.xpath(
            '//td[contains(text(), "Over 2.5")]/following-sibling::td[1]/text()').re_first(r'([\d.]+%)')
        item['under_2.5'] = response.xpath(
            '//td[contains(text(), "Under 2.5")]/following-sibling::td[1]/text()').re_first(r'([\d.]+%)')
        item['over_3.5'] = response.xpath(
            '//td[contains(text(), "Over 3.5")]/following-sibling::td[1]/text()').re_first(r'([\d.]+%)')
        item['under_3.5'] = response.xpath(
            '//td[contains(text(), "Under 3.5")]/following-sibling::td[1]/text()').re_first(r'([\d.]+%)')

        # Extract team goals
        item['home_over_0.5'] = response.xpath(
            '//td[contains(text(), "Arsenal Goals")]/following::td[contains(text(), "Over 0.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')
        item['home_under_0.5'] = response.xpath(
            '//td[contains(text(), "Arsenal Goals")]/following::td[contains(text(), "Under 0.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')
        item['home_over_1.5'] = response.xpath(
            '//td[contains(text(), "Arsenal Goals")]/following::td[contains(text(), "Over 1.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')
        item['home_under_1.5'] = response.xpath(
            '//td[contains(text(), "Arsenal Goals")]/following::td[contains(text(), "Under 1.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')

        item['away_over_0.5'] = response.xpath(
            '//td[contains(text(), "Paris Saint-Germain Goals")]/following::td[contains(text(), "Over 0.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')
        item['away_under_0.5'] = response.xpath(
            '//td[contains(text(), "Paris Saint-Germain Goals")]/following::td[contains(text(), "Under 0.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')
        item['away_over_1.5'] = response.xpath(
            '//td[contains(text(), "Paris Saint-Germain Goals")]/following::td[contains(text(), "Over 1.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')
        item['away_under_1.5'] = response.xpath(
            '//td[contains(text(), "Paris Saint-Germain Goals")]/following::td[contains(text(), "Under 1.5")]/following-sibling::td[1]/text()').re_first(
            r'([\d.]+%)')

        # Extract score analysis
        home_scores = response.css('td.homBG.lc_block::text').re(r'(\d+-\d+ @ [\d.]+%)')
        draw_scores = response.css('td.drwBG.lc_block::text').re(r'(\d+-\d+ @ [\d.]+%)')
        away_scores = response.css('td.awaBG.lc_block::text').re(r'(\d+-\d+ @ [\d.]+%)')

        item['home_most_likely_scores'] = home_scores
        item['draw_most_likely_scores'] = draw_scores
        item['away_most_likely_scores'] = away_scores

        print(item)

        def remove_null_or_none(data):
            if isinstance(data, dict):
                return {
                    k: remove_null_or_none(v)
                    for k, v in data.items()
                    if v not in (None, "null", "Null") and remove_null_or_none(v) is not None
                }
            elif isinstance(data, list):
                filtered_list = [remove_null_or_none(item) for item in data]
                return [item for item in filtered_list if item not in (None, "null", "Null")]
            elif data in (None, "null", "Null"):
                return None
            else:
                return data

        # Example usage:
        cleaned_stats = remove_null_or_none(item)

        yield {
            "match_time": match_time,
            "home_team": home_team,
            "away_team": away_team,
            "league_name": league_name,
            "combined_paragraph": combined_paragraph,
            "home_team_form": home_form,
            "away_team_form": away_form,
            "stats": cleaned_stats
        }





