# Padel Court Dimensions (in meters)
# Court is 20m long x 10m wide
COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
HALF_COURT_LENGTH = 10.0  # Each side is 10m

# Service line is 3m from the net
SERVICE_LINE_DISTANCE = 3.0

# For compatibility with existing code
SINGLE_LINE_WIDTH = COURT_WIDTH
DOUBLE_LINE_WIDTH = COURT_WIDTH
HALF_COURT_LINE_HEIGHT = HALF_COURT_LENGTH
SERVICE_LINE_WIDTH = SERVICE_LINE_DISTANCE

# Padel doesn't have doubles alley (enclosed court), but needed for code compatibility
DOUBLE_ALLY_DIFFERENCE = 0.0  # No alley in padel

# No man's land distance (service line to baseline area)
NO_MANS_LAND_HEIGHT = SERVICE_LINE_DISTANCE  # 3m from net to service line

# Player heights (average for 4 players)
PLAYER_1_HEIGHT_METERS = 1.80  # Team 1 - Player near
PLAYER_2_HEIGHT_METERS = 1.80  # Team 1 - Player far
PLAYER_3_HEIGHT_METERS = 1.80  # Team 2 - Player near
PLAYER_4_HEIGHT_METERS = 1.80  # Team 2 - Player far
