from scrapy import Request
from scrapy import Spider


def get_start_urls():
    start_url = 'https://www.avito.ru/moskva/kollektsionirovanie/monety'
    res = []
    res.append(start_url)
    for i in range(2, 726):
        res.append(start_url + '?p={}'.format(i))
    return res


class AvitoSpider(Spider):
    name = 'avito'
    allowed_domains = ['www.avito.ru']
    start_urls = get_start_urls()  # [:2]

    def parse(self, response):

        item_links = (
            response.selector
            .xpath("//a[@class='item-description-title-link']/@href")
            .extract())

        for link in item_links:
            yield Request(response.urljoin(link), callback=self.parse_item)

        # next_page_link = (
        #     response.selector
        #     .xpath("//a[contains(@class,'js-pagination-next')]/@href")
        #     .extract_first())

        # if next_page_link:
        #     yield Request(response.urljoin(next_page_link), self.parse)

    def parse_item(self, response):

        title = (response.selector
                 # .xpath("//meta[@property='og:title']/@content")
                 .xpath("//div[@class='sticky-header-prop sticky-header-title']//text()")
                 .extract_first())
        text = (response.selector
                # .xpath("//meta[@property='og:description']/@content")
                .xpath("//div[@itemprop='description']/p/text()")
                .extract())
        yield {'title': title, 'text': text, 'url': response.url}
