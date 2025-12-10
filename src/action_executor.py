"""
Action execution component for the Gesture Control System.
Executes OS-level actions based on recognized gestures.
"""
import subprocess
import logging
import platform
from typing import Optional
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController

from src.models import Action, ActionType


logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Executes OS-level actions triggered by gesture recognition.
    
    Supports launching applications, simulating keystrokes, media control,
    and system control operations.
    """
    
    def __init__(self):
        """Initialize the action executor with keyboard and mouse controllers."""
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.system = platform.system()  # 'Windows', 'Linux', or 'Darwin' (macOS)
        logger.info(f"ActionExecutor initialized for {self.system}")
    
    def execute(self, action: Action) -> bool:
        """
        Execute an OS action based on its type.
        
        Args:
            action: The Action object to execute
            
        Returns:
            bool: True if action executed successfully, False otherwise
        """
        try:
            action_type_str = action.type.value if hasattr(action.type, 'value') else str(action.type)
            logger.info(f"Executing action: {action_type_str} with data: {action.data}")
            
            if action.type == ActionType.LAUNCH_APP:
                return self._launch_app(action.data)
            elif action.type == ActionType.KEYSTROKE:
                return self._simulate_keystroke(action.data)
            elif action.type == ActionType.MEDIA_CONTROL:
                return self._media_control(action.data)
            elif action.type == ActionType.SYSTEM_CONTROL:
                return self._system_control(action.data)
            else:
                logger.error(f"Unknown action type: {action.type}")
                return False
                
        except Exception as e:
            action_type_str = action.type.value if hasattr(action.type, 'value') else str(action.type)
            logger.exception(f"Failed to execute action {action_type_str}: {e}")
            return False
    
    def _launch_app(self, data: dict) -> bool:
        """
        Launch an application using subprocess.
        
        Args:
            data: Dictionary containing 'path' key with application path
            
        Returns:
            bool: True if launch succeeded, False otherwise
        """
        try:
            app_path = data.get('path')
            if not app_path:
                logger.error("No application path provided")
                return False
            
            logger.info(f"Launching application: {app_path}")
            
            if self.system == 'Windows':
                subprocess.Popen(app_path, shell=True)
            elif self.system == 'Darwin':  # macOS
                subprocess.Popen(['open', app_path])
            else:  # Linux
                subprocess.Popen(app_path, shell=True)
            
            logger.info(f"Successfully launched: {app_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Application not found: {app_path}")
            return False
        except PermissionError:
            logger.error(f"Permission denied to launch: {app_path}")
            return False
        except Exception as e:
            logger.exception(f"Failed to launch application: {e}")
            return False
    
    def _simulate_keystroke(self, data: dict) -> bool:
        """
        Simulate keyboard input using pynput.
        
        Args:
            data: Dictionary containing 'keys' list with key names
            
        Returns:
            bool: True if keystroke succeeded, False otherwise
        """
        try:
            keys = data.get('keys', [])
            if not keys:
                logger.error("No keys provided for keystroke")
                return False
            
            logger.info(f"Simulating keystroke: {keys}")
            
            # Map string key names to pynput Key objects
            key_map = {
                'ctrl': Key.ctrl,
                'control': Key.ctrl,
                'alt': Key.alt,
                'shift': Key.shift,
                'cmd': Key.cmd,
                'command': Key.cmd,
                'win': Key.cmd,
                'windows': Key.cmd,
                'enter': Key.enter,
                'return': Key.enter,
                'space': Key.space,
                'tab': Key.tab,
                'backspace': Key.backspace,
                'delete': Key.delete,
                'esc': Key.esc,
                'escape': Key.esc,
                'up': Key.up,
                'down': Key.down,
                'left': Key.left,
                'right': Key.right,
                'home': Key.home,
                'end': Key.end,
                'page_up': Key.page_up,
                'page_down': Key.page_down,
                'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
                'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
                'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
            }
            
            # Convert string keys to Key objects or characters
            parsed_keys = []
            for key_str in keys:
                key_lower = key_str.lower()
                if key_lower in key_map:
                    parsed_keys.append(key_map[key_lower])
                elif len(key_str) == 1:
                    # Single character key
                    parsed_keys.append(key_str)
                else:
                    logger.warning(f"Unknown key: {key_str}")
                    parsed_keys.append(key_str)
            
            # Press all keys (for combinations)
            for key in parsed_keys:
                self.keyboard.press(key)
            
            # Release all keys in reverse order
            for key in reversed(parsed_keys):
                self.keyboard.release(key)
            
            logger.info(f"Successfully simulated keystroke: {keys}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to simulate keystroke: {e}")
            return False
    
    def _media_control(self, data: dict) -> bool:
        """
        Control media playback (play/pause/next/previous).
        
        Args:
            data: Dictionary containing 'command' key with media command
            
        Returns:
            bool: True if media control succeeded, False otherwise
        """
        try:
            command = data.get('command')
            if not command:
                logger.error("No media command provided")
                return False
            
            logger.info(f"Executing media control: {command}")
            
            # Map media commands to appropriate keys
            media_key_map = {
                'play_pause': Key.media_play_pause,
                'play': Key.media_play_pause,
                'pause': Key.media_play_pause,
                'next': Key.media_next,
                'previous': Key.media_previous,
                'prev': Key.media_previous,
                'stop': Key.media_play_pause,
                'volume_up': Key.media_volume_up,
                'volume_down': Key.media_volume_down,
                'mute': Key.media_volume_mute,
            }
            
            media_key = media_key_map.get(command.lower())
            if not media_key:
                logger.error(f"Unknown media command: {command}")
                return False
            
            # Press and release the media key
            self.keyboard.press(media_key)
            self.keyboard.release(media_key)
            
            logger.info(f"Successfully executed media control: {command}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to execute media control: {e}")
            return False
    
    def _system_control(self, data: dict) -> bool:
        """
        Execute system control operations (volume/lock/window switching).
        
        Args:
            data: Dictionary containing 'command' and optional parameters
            
        Returns:
            bool: True if system control succeeded, False otherwise
        """
        try:
            command = data.get('command')
            if not command:
                logger.error("No system command provided")
                return False
            
            logger.info(f"Executing system control: {command}")
            
            command_lower = command.lower()
            
            # Volume control
            if command_lower in ['volume_up', 'volume_down', 'volume_mute']:
                amount = data.get('amount', 1)
                return self._volume_control(command_lower, amount)
            
            # Lock screen
            elif command_lower == 'lock':
                return self._lock_screen()
            
            # Window switching
            elif command_lower in ['switch_window', 'alt_tab']:
                return self._switch_window()
            
            # Minimize window
            elif command_lower == 'minimize':
                return self._minimize_window()
            
            # Maximize window
            elif command_lower == 'maximize':
                return self._maximize_window()
            
            # Close window
            elif command_lower == 'close_window':
                return self._close_window()
            
            else:
                logger.error(f"Unknown system command: {command}")
                return False
                
        except Exception as e:
            logger.exception(f"Failed to execute system control: {e}")
            return False
    
    def _volume_control(self, command: str, amount: int = 1) -> bool:
        """Control system volume."""
        try:
            if command == 'volume_up':
                for _ in range(amount):
                    self.keyboard.press(Key.media_volume_up)
                    self.keyboard.release(Key.media_volume_up)
            elif command == 'volume_down':
                for _ in range(amount):
                    self.keyboard.press(Key.media_volume_down)
                    self.keyboard.release(Key.media_volume_down)
            elif command == 'volume_mute':
                self.keyboard.press(Key.media_volume_mute)
                self.keyboard.release(Key.media_volume_mute)
            
            logger.info(f"Volume control executed: {command}")
            return True
        except Exception as e:
            logger.exception(f"Volume control failed: {e}")
            return False
    
    def _lock_screen(self) -> bool:
        """Lock the screen based on OS."""
        try:
            if self.system == 'Windows':
                subprocess.run(['rundll32.exe', 'user32.dll,LockWorkStation'])
            elif self.system == 'Darwin':  # macOS
                subprocess.run(['/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession', '-suspend'])
            else:  # Linux
                # Try common lock commands
                try:
                    subprocess.run(['gnome-screensaver-command', '--lock'])
                except FileNotFoundError:
                    try:
                        subprocess.run(['xdg-screensaver', 'lock'])
                    except FileNotFoundError:
                        logger.error("No screen lock command found")
                        return False
            
            logger.info("Screen locked")
            return True
        except Exception as e:
            logger.exception(f"Failed to lock screen: {e}")
            return False
    
    def _switch_window(self) -> bool:
        """Switch between windows using Alt+Tab."""
        try:
            self.keyboard.press(Key.alt)
            self.keyboard.press(Key.tab)
            self.keyboard.release(Key.tab)
            self.keyboard.release(Key.alt)
            
            logger.info("Window switch executed")
            return True
        except Exception as e:
            logger.exception(f"Failed to switch window: {e}")
            return False
    
    def _minimize_window(self) -> bool:
        """Minimize the current window."""
        try:
            if self.system == 'Windows':
                self.keyboard.press(Key.cmd)
                self.keyboard.press(Key.down)
                self.keyboard.release(Key.down)
                self.keyboard.release(Key.cmd)
            elif self.system == 'Darwin':  # macOS
                self.keyboard.press(Key.cmd)
                self.keyboard.press('m')
                self.keyboard.release('m')
                self.keyboard.release(Key.cmd)
            else:  # Linux
                self.keyboard.press(Key.alt)
                self.keyboard.press(Key.f9)
                self.keyboard.release(Key.f9)
                self.keyboard.release(Key.alt)
            
            logger.info("Window minimized")
            return True
        except Exception as e:
            logger.exception(f"Failed to minimize window: {e}")
            return False
    
    def _maximize_window(self) -> bool:
        """Maximize the current window."""
        try:
            if self.system == 'Windows':
                self.keyboard.press(Key.cmd)
                self.keyboard.press(Key.up)
                self.keyboard.release(Key.up)
                self.keyboard.release(Key.cmd)
            elif self.system == 'Darwin':  # macOS
                # macOS doesn't have a standard maximize shortcut
                # This is a workaround using full screen
                self.keyboard.press(Key.ctrl)
                self.keyboard.press(Key.cmd)
                self.keyboard.press('f')
                self.keyboard.release('f')
                self.keyboard.release(Key.cmd)
                self.keyboard.release(Key.ctrl)
            else:  # Linux
                self.keyboard.press(Key.alt)
                self.keyboard.press(Key.f10)
                self.keyboard.release(Key.f10)
                self.keyboard.release(Key.alt)
            
            logger.info("Window maximized")
            return True
        except Exception as e:
            logger.exception(f"Failed to maximize window: {e}")
            return False
    
    def _close_window(self) -> bool:
        """Close the current window."""
        try:
            if self.system == 'Windows':
                self.keyboard.press(Key.alt)
                self.keyboard.press(Key.f4)
                self.keyboard.release(Key.f4)
                self.keyboard.release(Key.alt)
            elif self.system == 'Darwin':  # macOS
                self.keyboard.press(Key.cmd)
                self.keyboard.press('w')
                self.keyboard.release('w')
                self.keyboard.release(Key.cmd)
            else:  # Linux
                self.keyboard.press(Key.alt)
                self.keyboard.press(Key.f4)
                self.keyboard.release(Key.f4)
                self.keyboard.release(Key.alt)
            
            logger.info("Window closed")
            return True
        except Exception as e:
            logger.exception(f"Failed to close window: {e}")
            return False
