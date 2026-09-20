import logging
import re
import urllib.parse as urlparse
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from io import StringIO
from typing import List, Optional

import aiohttp

from moodle_dl.types import MoodleURL
from moodle_dl.utils import MoodleDLCookieJar, convert_to_aiohttp_cookie_jar

#: Query parameters that carry Moodle credentials and must never leave the Moodle origin
TOKEN_QUERY_PARAMS = ('token', 'wstoken')

#: HTTP status codes that carry a Location header and therefore trigger a redirect
REDIRECT_STATUSES = {301, 302, 303, 307, 308}

#: Maximum number of redirect hops we follow manually
MAX_REDIRECT_HOPS = 10

DEFAULT_PORTS = {'http': 80, 'https': 443}


def censor_url(url: str) -> str:
    """Remove token values from a URL before it is written to a log file."""
    if url is None:
        return url
    return re.sub(r'((?:ws)?token)=([^&#]*)', r'\1=censored', str(url))


@dataclass
class RedirectChain:
    """
    The three different URLs a download can be based on:
    @param original_url: The link exactly as Moodle reported it (never modified)
    @param request_url: The sanitized link that is actually requested
                        (token only attached on same-origin Moodle install paths)
    @param final_url: The link after following all redirects
    @param history: All sanitized links visited while following redirects (incl. final_url)
    @param crossed_origin: True if a redirect left the configured Moodle origin
    """

    original_url: str
    request_url: str
    final_url: str
    history: List[str] = field(default_factory=list)
    crossed_origin: bool = False


class RedirectBlockedError(Exception):
    """Gets raised if a redirect must not be followed by the security policy."""

    def __init__(self, reason: str, target_url: str = None):
        message = f'Redirect blocked by policy: {reason}'
        if target_url is not None:
            message += f' (target: {censor_url(target_url)})'
        super().__init__(message)
        self.reason = reason
        self.target_url = target_url


class RedirectPolicy:
    """
    Redirect security policy rooted at the configured Moodle scheme/domain/path.

    * The Moodle token is only attached to the initial Moodle URL and to URLs on the
      same-origin Moodle installation path. On a cross-origin redirect it is stripped.
    * Moodle cookies are only sent on same-origin requests (see SameOriginCookieJar).
    * Every redirect hop is recorded and inspected. A cross-origin redirect is only
      followed if the target host is on the explicit trusted redirect domain list
      (and is not rejected by the existing domain black/white list).
    """

    def __init__(
        self,
        moodle_url: MoodleURL,
        token: Optional[str],
        trusted_redirect_domains: List[str] = None,
        domains_whitelist: List[str] = None,
        domains_blacklist: List[str] = None,
    ):
        self.token = token
        self.scheme = moodle_url.scheme.rstrip(':/')
        root = urlparse.urlparse(moodle_url.url_base)
        self.moodle_host = (root.hostname or '').lower()
        self.moodle_port = root.port or DEFAULT_PORTS.get(self.scheme)
        install_path = moodle_url.path or '/'
        if not install_path.startswith('/'):
            install_path = '/' + install_path
        self.install_path = install_path

        self.trusted_redirect_domains = [d.lower() for d in (trusted_redirect_domains or []) if d]
        self.domains_whitelist = [d.lower() for d in (domains_whitelist or []) if d]
        self.domains_blacklist = [d.lower() for d in (domains_blacklist or []) if d]

    # ------------------------------ origin checks ----------------------------

    def _origin(self, url: str):
        parsed = urlparse.urlparse(str(url))
        scheme = (parsed.scheme or '').lower()
        host = (parsed.hostname or '').lower()
        port = parsed.port or DEFAULT_PORTS.get(scheme)
        return scheme, host, port, parsed.path or ''

    def is_same_origin(self, url: str) -> bool:
        """True if the URL shares scheme and host with the Moodle installation."""
        scheme, host, port, _path = self._origin(url)
        return scheme == self.scheme and host == self.moodle_host and port == self.moodle_port

    def is_install_path(self, url: str) -> bool:
        """True if the URL is same-origin and located below the Moodle install path."""
        scheme, host, port, path = self._origin(url)
        if scheme != self.scheme or host != self.moodle_host or port != self.moodle_port:
            return False
        return path == self.install_path.rstrip('/') or path.startswith(self.install_path)

    @staticmethod
    def _domain_matches(host: str, port, entry: str) -> bool:
        # Entries may also contain an explicit port ("video.example.edu:8443")
        if ':' in entry:
            entry_host, entry_port = entry.rsplit(':', 1)
            if entry_port.isdigit() and (port is None or int(entry_port) != port):
                return False
            entry = entry_host
        return host == entry or host.endswith('.' + entry)

    def is_trusted_redirect_domain(self, host: str, port=None) -> bool:
        host = (host or '').lower()
        return any(self._domain_matches(host, port, entry) for entry in self.trusted_redirect_domains)

    def is_blacklisted_domain(self, host: str, port=None) -> bool:
        host = (host or '').lower()
        return any(self._domain_matches(host, port, entry) for entry in self.domains_blacklist)

    def is_whitelisted_domain(self, host: str, port=None) -> bool:
        # An empty whitelist means "all domains are allowed"
        if len(self.domains_whitelist) == 0:
            return True
        host = (host or '').lower()
        return any(self._domain_matches(host, port, entry) for entry in self.domains_whitelist)

    # ------------------------------ URL sanitizing ---------------------------

    @staticmethod
    def _modify_query(parsed, *, attach_token, token):
        query_pairs = [
            (key, value)
            for key, value in urlparse.parse_qsl(parsed.query, keep_blank_values=True)
            if key not in TOKEN_QUERY_PARAMS
        ]
        if attach_token and token:
            query_pairs.append(('token', token))
        return urlparse.urlencode(query_pairs, doseq=True)

    def sanitize_url(self, url: str, attach_token: bool) -> str:
        """
        Build the URL that may actually be requested.
        The token is only attached on same-origin Moodle install paths;
        any pre-existing token is stripped from every other URL.
        """
        if url is None or url == '' or url.startswith('data:'):
            return url

        parsed = urlparse.urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return url

        allow_token = attach_token and self.is_install_path(url) and bool(self.token)
        new_query = self._modify_query(parsed, attach_token=allow_token, token=self.token)
        return urlparse.urlunparse(parsed._replace(query=new_query))

    # ------------------------------ redirect hops ----------------------------

    def evaluate_redirect(
        self, current_url: str, target_url: str, attach_token: bool, enforce_domain_filter: bool
    ) -> str:
        """
        Decide how a redirect from current_url to target_url has to be handled.
        @return: The sanitized URL that may be requested next.
        @raise RedirectBlockedError: If the redirect must not be followed.
        """
        parsed = urlparse.urlparse(target_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise RedirectBlockedError(f'Redirect target uses unsupported scheme {parsed.scheme!r}', target_url)

        target_host = (parsed.hostname or '').lower()
        target_port = parsed.port or DEFAULT_PORTS.get((parsed.scheme or '').lower())

        if self.is_same_origin(target_url):
            # Same-origin Moodle hop: cookies are allowed, token handling is unchanged.
            return self.sanitize_url(target_url, attach_token)

        # The redirect leaves the Moodle origin: the token is stripped and Moodle
        # cookies are withheld (the cookie jar refuses them on cross-origin requests).
        sanitized = self.sanitize_url(target_url, False)

        if not self.is_trusted_redirect_domain(target_host, target_port):
            raise RedirectBlockedError(
                f'Cross-origin redirect target {target_host!r} is not in the list of trusted redirect domains',
                target_url,
            )

        if self.is_blacklisted_domain(target_host, target_port):
            raise RedirectBlockedError(
                f'Cross-origin redirect target {target_host!r} is on the download domain blacklist', target_url
            )

        if enforce_domain_filter and not self.is_whitelisted_domain(target_host, target_port):
            raise RedirectBlockedError(
                f'Cross-origin redirect target {target_host!r} is not on the download domain whitelist', target_url
            )

        return sanitized

    async def open(
        self,
        session: aiohttp.ClientSession,
        method: str,
        request_url: str,
        *,
        headers=None,
        ssl=None,
        timeout=None,
        attach_token: bool,
        enforce_domain_filter: bool,
    ):
        """
        Opens request_url and follows redirects manually so that the policy can be
        enforced on every single hop.
        @return: Tuple of the final aiohttp response (caller must consume/release it)
                 and the RedirectChain that describes the followed path.
        """
        original_url = request_url
        current_url = self.sanitize_url(request_url, attach_token)
        # The sanitized initial URL is a real URL (usable for requests);
        # history entries are only used in logs and are censored.
        sanitized_request_url = current_url
        history: List[str] = []
        crossed_origin = False
        current_method = method

        hops = 0
        while True:
            if hops > MAX_REDIRECT_HOPS:
                raise RedirectBlockedError('Too many redirects', current_url)

            response = await session.request(
                current_method,
                current_url,
                headers=headers,
                ssl=ssl,
                timeout=timeout,
                allow_redirects=False,
            )

            visited_url = censor_url(str(response.url))
            history.append(visited_url)

            if response.status not in REDIRECT_STATUSES or 'Location' not in response.headers:
                return (
                    response,
                    RedirectChain(
                        original_url=original_url,
                        request_url=sanitized_request_url,
                        final_url=str(response.url),
                        history=history,
                        crossed_origin=crossed_origin,
                    ),
                )

            location = response.headers.get('Location', '')
            target_url = urlparse.urljoin(str(response.url), location)
            await response.release()

            if not self.is_same_origin(target_url):
                crossed_origin = True
                logging.debug(
                    'Cross-origin redirect detected: %s -> %s; '
                    'Moodle token is stripped and Moodle cookies are withheld',
                    censor_url(current_url),
                    censor_url(target_url),
                )

            next_url = self.evaluate_redirect(current_url, target_url, attach_token, enforce_domain_filter)
            if next_url != target_url:
                logging.debug('Redirect target was sanitized to: %s', censor_url(next_url))

            if response.status == 303 and current_method != 'HEAD':
                current_method = 'GET'

            logging.debug(
                'Following redirect (%d/%d): %s -> %s',
                hops + 1,
                MAX_REDIRECT_HOPS,
                censor_url(current_url),
                censor_url(target_url),
            )
            current_url = next_url
            hops += 1

    # ------------------------------ cookies ----------------------------------

    def new_cookie_jar(self, cookies_text: Optional[str]) -> 'SameOriginCookieJar':
        """
        Build the cookie jar used for a download session.
        Moodle cookies are stored separately and are never attached to cross-origin
        requests, even if a redirect tries to forward them.
        """
        moodle_jar = aiohttp.CookieJar(unsafe=True)
        if cookies_text is not None:
            file_jar = MoodleDLCookieJar(StringIO(cookies_text))
            file_jar.load(ignore_discard=True, ignore_expires=True)
            moodle_jar = convert_to_aiohttp_cookie_jar(file_jar)
        return SameOriginCookieJar(moodle_jar, self.is_same_origin)


class SameOriginCookieJar(aiohttp.CookieJar):
    """
    Cookie jar that keeps Moodle cookies and foreign cookies in separate jars.
    Moodle cookies are only returned for same-origin requests; cookies set by a
    foreign host are only returned for that foreign host.
    """

    def __init__(self, moodle_jar: aiohttp.CookieJar, is_same_origin):
        super().__init__(unsafe=True)
        self._moodle_jar = moodle_jar
        self._is_same_origin = is_same_origin

    def filter_cookies(self, request_url):
        foreign_cookies = super().filter_cookies(request_url)
        if not self._is_same_origin(str(request_url)):
            # Cross-origin request: only cookies collected from foreign hosts may be sent.
            return foreign_cookies
        # Same-origin request: Moodle cookies plus cookies collected from the Moodle host.
        merged = SimpleCookie()
        for morsel in self._moodle_jar.filter_cookies(request_url).values():
            merged[morsel.key] = morsel
        for morsel in foreign_cookies.values():
            merged[morsel.key] = morsel
        return merged

    def update_cookies(self, cookies, response_url=None):
        if response_url is not None and self._is_same_origin(str(response_url)):
            self._moodle_jar.update_cookies(cookies, response_url)
        else:
            super().update_cookies(cookies, response_url)

    def clear(self, predicate=None):
        super().clear(predicate)
        self._moodle_jar.clear(predicate)

    def clear_domain(self, domain):
        super().clear_domain(domain)
        self._moodle_jar.clear_domain(domain)
