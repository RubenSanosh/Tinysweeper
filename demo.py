"""Real-time Pygame visualization of TinySweeper playing 6x6/4 Minesweeper."""

import warnings
import pygame
import torch
import torch.nn as nn

from env import MinesweeperEnv
from train import TinyLinearConvQ, select_action

warnings.filterwarnings("ignore", category=DeprecationWarning)

WIN_SIZE = 400
TOP_BAR = 56
GRID_PAD = 16
FPS = 30

BG = (18, 20, 26)
BAR_BG = (28, 31, 40)
CELL_HIDDEN = (46, 52, 68)
CELL_REVEALED = (30, 34, 44)
CELL_MINE = (210, 64, 72)
BORDER = (20, 22, 28)
TEXT_MAIN = (235, 240, 250)
TEXT_DIM = (150, 160, 180)
NUM_COLORS = {
    1: (98, 175, 239),
    2: (116, 204, 130),
    3: (235, 110, 110),
    4: (182, 132, 238),
    5: (230, 162, 92),
    6: (86, 198, 198),
    7: (235, 235, 235),
    8: (160, 160, 160),
}


def load_tiny_model(path="model_tiny.pth"):
    """Load the INT8 dynamically quantized Q-network and compute parameter count."""
    base = TinyLinearConvQ(16)
    param_count = sum(p.numel() for p in base.parameters())
    qnet = torch.ao.quantization.quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
    # Quantized Linear packed params require weights_only=False on PyTorch 2.6+.
    qnet.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    qnet.eval()
    return qnet, param_count


def draw_board(screen, font_bar, font_cell, env, games, wins, param_count):
    screen.fill(BG)
    pygame.draw.rect(screen, BAR_BG, (0, 0, WIN_SIZE, TOP_BAR))

    win_rate = (wins / games * 100.0) if games > 0 else 0.0
    left_txt = font_bar.render(f"Win Rate: {win_rate:.0f}%", True, TEXT_MAIN)
    right_txt = font_bar.render(f"Model: {param_count:,} params", True, TEXT_DIM)
    screen.blit(left_txt, (14, 18))
    screen.blit(right_txt, (WIN_SIZE - right_txt.get_width() - 14, 18))

    board_px = WIN_SIZE - 2 * GRID_PAD
    cell_px = board_px // env.SIZE
    y0 = TOP_BAR + (WIN_SIZE - TOP_BAR - cell_px * env.SIZE) // 2
    x0 = (WIN_SIZE - cell_px * env.SIZE) // 2

    for r in range(env.SIZE):
        for c in range(env.SIZE):
            rect = pygame.Rect(
                x0 + c * cell_px + 2, y0 + r * cell_px + 2, cell_px - 4, cell_px - 4
            )
            if not env.revealed[r, c]:
                pygame.draw.rect(screen, CELL_HIDDEN, rect, border_radius=6)
            elif env.mines[r, c]:
                pygame.draw.rect(screen, CELL_MINE, rect, border_radius=6)
                lbl = font_cell.render("*", True, TEXT_MAIN)
                screen.blit(lbl, lbl.get_rect(center=rect.center))
            else:
                pygame.draw.rect(screen, CELL_REVEALED, rect, border_radius=6)
                n = int(env.counts[r, c])
                if n > 0:
                    lbl = font_cell.render(str(n), True, NUM_COLORS.get(n, TEXT_MAIN))
                    screen.blit(lbl, lbl.get_rect(center=rect.center))
            pygame.draw.rect(screen, BORDER, rect, 1, border_radius=6)

    pygame.display.flip()


def main(max_frames=None):
    pygame.init()
    screen = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
    pygame.display.set_caption("TinySweeper")
    clock = pygame.time.Clock()
    font_bar = pygame.font.SysFont("monospace", 16, bold=True)
    font_cell = pygame.font.SysFont("monospace", 24, bold=True)

    qnet, param_count = load_tiny_model("model_tiny.pth")
    env = MinesweeperEnv()
    obs, _ = env.reset()

    games, wins = 0, 0
    step_cooldown = 0  # Pace clicks slightly so moves are readable at 30 FPS.
    pause_frames = 0
    frames = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    games, wins = 0, 0
                    pause_frames = 0

        if pause_frames > 0:
            pause_frames -= 1
            if pause_frames == 0:
                obs, _ = env.reset()
        elif step_cooldown > 0:
            step_cooldown -= 1
        else:
            action = select_action(qnet, obs)
            obs, _, terminated, truncated, info = env.step(action)
            step_cooldown = 4
            if terminated or truncated:
                games += 1
                if info.get("won"):
                    wins += 1
                pause_frames = 12

        draw_board(screen, font_bar, font_cell, env, games, wins, param_count)
        clock.tick(FPS)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
