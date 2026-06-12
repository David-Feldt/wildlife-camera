import pytest

from crittercam.config import Config, EventsConfig


@pytest.fixture
def cfg(tmp_path):
    return Config(
        data_root=tmp_path,
        events=EventsConfig(min_track_frames=3, linger_seconds=1.0,
                            preroll_seconds=2.0, max_clip_seconds=8.0),
    )
