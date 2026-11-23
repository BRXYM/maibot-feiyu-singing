"""网易云歌曲分享的公开资料与原音频。"""
from functools import lru_cache
from urllib.parse import parse_qs, urljoin, urlsplit
import re

import httpx


def song_id(url):
    p = urlsplit(url)
    if p.scheme not in ('http', 'https') or p.netloc.lower() not in ('music.163.com', 'y.music.163.com'):
        return None
    route = urlsplit(p.fragment) if p.fragment.startswith('/song?') else p
    ids = parse_qs(route.query).get('id', [])
    if route.path.rstrip('/') in ('/song', '/m/song') and len(ids) == 1 and re.fullmatch(r'[1-9]\d{0,17}', ids[0]):
        return ids[0]
    return None


@lru_cache(maxsize=128)
def track_info(identifier):
    if not re.fullmatch(r'[1-9]\d{0,17}', identifier):
        raise ValueError('无效的网易云歌曲编号')
    try:
        with httpx.Client(timeout=3, trust_env=False, headers={'Referer': 'https://music.163.com/'}) as client:
            res = client.get('https://music.163.com/api/song/detail', params={'id': identifier, 'ids': '['+identifier+']'})
            res.raise_for_status()
            tracks = res.json().get('songs') or []
        track = next((s for s in tracks if str(s.get('id')) == identifier), None)
        if not track or not track.get('name') or not track.get('artists'):
            raise ValueError('网易云没有返回这首歌的资料')
        return dict(title=track['name'], artist=' / '.join(a['name'] for a in track['artists']),
                    duration=track['duration']/1000, album=track.get('album', {}).get('name', ''),
                    source_url='https://music.163.com/song?id='+identifier)
    except (httpx.HTTPError, KeyError, TypeError) as exc:
        raise ValueError('未能读取网易云原链接的歌曲资料') from exc


def download_track(identifier, path):
    url = 'https://music.163.com/song/media/outer/url?id=' + identifier + '.mp3'
    with httpx.Client(timeout=25, trust_env=False, headers={'Referer':'https://music.163.com/'}) as client:
        for _ in range(4):
            with client.stream('GET', url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers['location'])
                    p = urlsplit(url)
                    if p.scheme not in ('http', 'https') or not (p.hostname or '').endswith('.music.126.net') or p.username or p.password or p.port not in (None,80,443):
                        raise ValueError('网易云未提供这首歌的公开音频')
                    url = p._replace(scheme='https').geturl()
                    continue
                response.raise_for_status()
                if 'text/' in response.headers.get('content-type', ''):
                    raise ValueError('网易云返回网页，未提供公开原声')
                with path.open('wb') as dest:
                    for block in response.iter_bytes(65536):
                        if dest.tell()+len(block) > 30*1024**2:
                            raise ValueError('公开音轨超过下载大小上限')
                        dest.write(block)
                return
    raise ValueError('网易云音频跳转次数超限')
