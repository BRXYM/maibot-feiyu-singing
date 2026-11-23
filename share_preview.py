"""只读取当前 QQ 私聊最近分享的公开链接。"""
import asyncio
import ipaddress
import re
import socket
import time
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup
from yarl import URL

from .music_links import song_id, track_info


def shared_link(rows, sender):
    if not isinstance(rows, list):
        return None
    recent = sorted((r for r in rows if r.get('message_id') and
                     str(r.get('message_info', {}).get('user_info', {}).get('user_id')) == sender
                     and 0 <= time.time()-float(r['timestamp']) < 300), key=lambda r: float(r['timestamp']), reverse=True)
    if not recent:
        return None
    row = recent[0]
    if row.get('is_notify') or str(row['message_id']).startswith('qq-notice-'):
        return None
    if len(recent) > 1 and re.fullmatch(r'(?:你)?(?:这个|这首歌?|这条|这个链接|链接)(?:你)?(?:看了吗|能唱吗|可以唱吗|讲了什么|说了什么|是什么|呢)[？?。！! ]*', row.get('processed_plain_text','').strip()):
        row = recent[1]
    urls = re.findall(r'https?://[^\s<>"\u3000-\u303f\uff00-\uffef]+',row.get('processed_plain_text',''))
    if urls:
        return str(row['message_id']), urls[0].rstrip(').,;]！。')
    return None


def public_ip(value):
    addr = ipaddress.ip_address(value)
    return addr.is_global and not addr.is_multicast and not getattr(addr, 'scope_id', None)


class PublicResolver(aiohttp.resolver.ThreadedResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        addresses = await super().resolve(host, port, family)
        if not addresses or not all(public_ip(r['host']) for r in addresses):
            raise ValueError('链接不是公开网页')
        return addresses


async def page_preview(url):
    connector = aiohttp.TCPConnector(resolver=PublicResolver(), family=socket.AF_INET)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=3),
                                    cookie_jar=aiohttp.DummyCookieJar(), trust_env=False, auto_decompress=False,
                                    headers={'Accept-Encoding':'identity','Accept':'text/html,text/plain'}) as client:
        for _ in range(4):
            target = URL(url)
            host = target.raw_host
            if target.scheme not in ('http','https') or not host or target.user is not None or target.password is not None or target.port not in (80,443):
                raise ValueError('仅支持公开网页链接')
            if ':' in host or re.fullmatch(r'[.\d]+',host):
                if not public_ip(host):
                    raise ValueError('链接不是公开网页')
            async with client.get(target, allow_redirects=False) as res:
                if res.status in (301,302,303,307,308):
                    url = urljoin(url, res.headers['Location'])
                    continue
                res.raise_for_status()
                if res.content_type not in ('text/html','application/xhtml+xml','text/plain') or res.headers.get('Content-Encoding','identity') != 'identity':
                    raise ValueError('链接没有提供可读取的网页文本')
                data = bytearray()
                async for block in res.content.iter_chunked(16384):
                    data.extend(block)
                    if len(data) >= 256*1024:
                        break
                if res.content_type == 'text/plain':
                    return {'text': data.decode(res.charset or 'utf-8', errors='replace').strip()[:800]}
                doc = BeautifulSoup(bytes(data), 'html.parser', from_encoding=res.charset)
                for tag in doc.select('script,style,nav,footer,header,noscript,form,template'):
                    tag.decompose()
                title = doc.title.get_text(' ',strip=True)[:140] if doc.title else ''
                meta = doc.select_one('meta[property="og:description"],meta[name="description"]')
                desc = (meta.get('content') or '')[:360] if meta else ''
                body = doc.select_one('article,main') or doc.body
                paragraphs = body.select('p') if body else []
                excerpt = (' '.join(n.get_text(' ',strip=True) for n in paragraphs) or (body.get_text(' ',strip=True) if body else ''))[:800]
                if not (title or desc or excerpt):
                    raise ValueError('网页没有公开的可读正文')
                return {'title':title,'description':desc,'excerpt':excerpt}
    raise ValueError('网页跳转次数超限')


async def preview(url):
    identifier = song_id(url)
    if identifier:
        return await asyncio.to_thread(track_info, identifier)
    return await page_preview(url)
