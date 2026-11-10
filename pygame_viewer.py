"""Optional Pygame viewer. Importing this module does not require Pygame."""
import os


class ViewerClosed(Exception):
    """The user requested a clean stop via the window or Escape."""


class PygameViewer:
    def __init__(self, size, fps=30):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Visual mode requires pygame: python -m pip install pygame") from exc
        self.pg = pygame
        self.fps = fps
        self.paused = False
        self.cell = max(1, 560 // size)
        self.board_size = self.cell * size
        self.origin = ((760 - self.board_size) // 2, 152)
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((760, self.board_size + 248))
            pygame.display.set_caption("Auraboros | Snake Q-learning")
            self.font = pygame.font.Font(None, 25)
            self.small = pygame.font.Font(None, 22)
            self.clock = pygame.time.Clock()
        except pygame.error as exc:
            pygame.quit()
            raise RuntimeError(f"Could not open the Pygame viewer: {exc}") from exc

    def poll(self):
        """Process window controls, including during undisplayed training games."""
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                raise ViewerClosed()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pg.K_UP:
                    self.fps = min(240, max(self.fps + 1, round(self.fps * 1.5)))
                elif event.key == pg.K_DOWN:
                    self.fps = max(1, round(self.fps / 1.5))

    def _text(self, text, y, color=(215, 225, 238), small=False):
        font = self.small if small else self.font
        self.screen.blit(font.render(text, True, color), (24, y))

    def show(self, env, title, epsilon, total_reward, reward=None, action=None,
             summary="", reason=None, throttle=True):
        pg = self.pg
        # Keep the terminal board visible briefly, while still processing controls.
        end_at = pg.time.get_ticks() + (350 if reason else 0)
        while True:
            self.poll()
            self.screen.fill((17, 24, 39))
            self._text(title, 16)
            self._text(f"Apples {env.score}  Length {len(env.snake)}  Total steps {env.steps}"
                       f"  No food {env.steps_since_apple}/{env.timeout:g}", 46)
            last = "--" if reward is None else f"{reward:+.2f}"
            self._text(f"Epsilon {epsilon:.3f}   Return {total_reward:.2f}   Last reward {last}", 74)
            self._text(summary, 102, small=True)
            self._text(env.features.format_state(env.get_state()), 126, small=True)

            ox, oy = self.origin
            for row in range(env.size):
                for col in range(env.size):
                    rect = pg.Rect(ox + col * self.cell, oy + row * self.cell, self.cell, self.cell)
                    color = (30, 41, 59) if (row + col) % 2 else (34, 47, 65)
                    pg.draw.rect(self.screen, color, rect)
            for i, (row, col) in enumerate(env.snake):
                rect = pg.Rect(ox + col * self.cell, oy + row * self.cell, self.cell, self.cell)
                inset = min(2, self.cell // 4)
                pg.draw.rect(self.screen, (134, 239, 172) if i == 0 else (34, 197, 94),
                             rect.inflate(-2 * inset, -2 * inset), border_radius=min(6, self.cell // 4))
            if env.apple is not None:
                row, col = env.apple
                center = (ox + col * self.cell + self.cell // 2, oy + row * self.cell + self.cell // 2)
                pg.draw.circle(self.screen, (248, 113, 113), center, max(1, self.cell // 3))

            footer = oy + self.board_size + 16
            status = "PAUSED" if self.paused else (f"END: {reason}" if reason else "RUNNING")
            move = ("Straight", "Left", "Right")[action] if action is not None else "--"
            self._text(f"{status}   Action {move}   Speed {self.fps} steps/s", footer)
            self._text("Space: pause/resume   Up/Down: speed   Esc/close: stop", footer + 30, small=True)
            pg.display.flip()
            self.clock.tick(60 if self.paused else (self.fps if throttle else 0))
            if not self.paused and pg.time.get_ticks() >= end_at:
                return

    def close(self):
        self.pg.quit()
