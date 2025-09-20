import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from PIL import Image

from constants import HTTP_CACHE_DIR
from topography import async_fetch_terrain_rgb_tile, decode_terrain_rgb_to_elevation_m


@dataclass(frozen=True)
class TileKey:
    z: int
    x: int
    y: int
    retina: bool

    def path_parts(self) -> tuple[str, str, str]:
        return (
            str(self.z),
            str(self.x),
            f'{self.y}{"@2x" if self.retina else ""}.pngraw',
        )


class ElevationTileProvider:
    """
    Shared provider for Terrain-RGB tiles with:
    - in-flight request coalescing
    - in-memory cache of raw bytes (PIL images constructed on demand)
    - on-disk cache of raw bytes under HTTP_CACHE_DIR/terrain_rgb/z/x/y(@2x).pngraw.

    It exposes two async methods:
      - get_tile_image: returns PIL.Image (RGB) of a terrain tile
      - get_tile_dem: returns list[list[float]] DEM decoded from the tile
    """

    def __init__(
        self,
        client: aiohttp.ClientSession,
        api_key: str,
        *,
        use_retina: bool,
        cache_root: Path | None = None,
        max_mem_tiles: int = 512,
    ) -> None:
        self.client = client
        self.api_key = api_key
        self.use_retina = bool(use_retina)
        self.cache_root = (cache_root or Path(HTTP_CACHE_DIR)) / 'terrain_rgb'
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._inflight: dict[TileKey, asyncio.Future[bytes]] = {}
        self._mem_raw: dict[TileKey, bytes] = {}
        self._mem_lru: list[TileKey] = []
        self._mem_dem: dict[TileKey, list[list[float]]] = {}
        self._max_mem = max(16, int(max_mem_tiles))

    def _key(self, z: int, x: int, y: int) -> TileKey:
        return TileKey(int(z), int(x), int(y), self.use_retina)

    def _disk_path(self, key: TileKey) -> Path:
        z, x, y = key.path_parts()
        return self.cache_root / z / x / y

    def _remember_raw(self, key: TileKey, data: bytes) -> None:
        self._mem_raw[key] = data
        self._mem_lru.append(key)
        if len(self._mem_lru) > self._max_mem:
            old = self._mem_lru.pop(0)
            self._mem_raw.pop(old, None)
            self._mem_dem.pop(old, None)

    def _touch(self, key: TileKey) -> None:
        if key in self._mem_lru:
            with contextlib.suppress(ValueError):
                self._mem_lru.remove(key)
            self._mem_lru.append(key)

    async def _fetch_raw(self, key: TileKey) -> bytes:
        # 1) in-flight
        fut = self._inflight.get(key)
        if fut is not None:
            return await fut
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._inflight[key] = fut

        async def run() -> None:
            try:
                # mem cache
                if key in self._mem_raw:
                    self._touch(key)
                    fut.set_result(self._mem_raw[key])
                    return
                # disk cache
                p = self._disk_path(key)
                if p.exists():
                    data = p.read_bytes()
                    self._remember_raw(key, data)
                    fut.set_result(data)
                    return
                # network
                img = await async_fetch_terrain_rgb_tile(
                    client=self.client,
                    api_key=self.api_key,
                    z=key.z,
                    x=key.x,
                    y=key.y,
                    use_retina=key.retina,
                )
                # Save raw bytes; PIL Image has no direct raw, so re-save as PNG to bytes
                # However async_fetch_terrain_rgb_tile already reads PNGRAW bytes then constructs Image.
                # We'll rebuild bytes from the provided image to persist the payload.
                from io import BytesIO

                buf = BytesIO()
                img.save(buf, format='PNG')
                data = buf.getvalue()
                # ensure path exists
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                self._remember_raw(key, data)
                fut.set_result(data)
            except Exception as e:  # pragma: no cover - best effort
                fut.set_exception(e)
            finally:
                self._inflight.pop(key, None)

        asyncio.create_task(run())
        return await fut

    async def get_tile_image(self, z: int, x: int, y: int) -> Image.Image:
        key = self._key(z, x, y)
        # Try memory first
        raw = self._mem_raw.get(key)
        if raw is None:
            raw = await self._fetch_raw(key)
        else:
            self._touch(key)
        from io import BytesIO

        return Image.open(BytesIO(raw)).convert('RGB')

    async def get_tile_dem(self, z: int, x: int, y: int) -> list[list[float]]:
        key = self._key(z, x, y)
        dem = self._mem_dem.get(key)
        if dem is not None:
            self._touch(key)
            return dem
        img = await self.get_tile_image(z, x, y)
        dem2 = decode_terrain_rgb_to_elevation_m(img)
        self._mem_dem[key] = dem2
        return dem2
