from enum import Enum

class GameState(Enum):
  NOT_RUNNING = -1
  WAITING = 0
  EDITING_MAP = 1
  PLAYING = 2
  GAME_SHUTDOWN = 3
  SONG_SELECT_EDIT = 4
  SONG_SELECT = 5
  IDK = 6 # No one knows what this game state is lol.
  RESULTS = 7

class GameMode(Enum):
  STANDARD = 0
  TAIKO = 1
  CATCH = 2
  MANIA = 3
