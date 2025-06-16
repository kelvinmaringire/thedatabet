import scrapy


class ForebetSpider(scrapy.Spider):
    name = "forebet"

    def start_requests(self):


        yield scrapy.Request(
            data='league_url',
            callback=self.parse
        )


    def parse(self, response):
        pass
