import pyautogui
# import Quartz
import time
from PIL import Image

def find_window_by_title(title_substring):
    options = Quartz.kCGWindowListOptionOnScreenOnly
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    
    for window in window_list:
        window_name = window.get('kCGWindowName', '')
        owner_name = window.get('kCGWindowOwnerName', '')
        if title_substring.lower() in window_name.lower() or title_substring.lower() in owner_name.lower():
            bounds = window['kCGWindowBounds']
            x = int(bounds['X'])
            y = int(bounds['Y'])
            width = int(bounds['Width'])
            height = int(bounds['Height'])
            return (x, y, x+width, y+height)
    
    print('Window names:', [window.get('kCGWindowName', '') for window in window_list])
    raise RuntimeError(f"No window found with title including '{title_substring}'")

def get_screen_scale_factor():
    screen_size_points = pyautogui.size()  # in points (logical)
    screenshot_size_pixels = pyautogui.screenshot().size  # in actual pixels
    scale_x = screenshot_size_pixels[0] / screen_size_points[0]
    scale_y = screenshot_size_pixels[1] / screen_size_points[1]
    return (scale_x, scale_y)