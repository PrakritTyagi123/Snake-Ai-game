import argparse, os, random, sys, time
import numpy as np
import pygame

from constants import GRID_W, GRID_H, FPS_MIN, FPS_INIT, BLINK_DUR
from dqn_agent import DQNAgent

# ── colours ──────────────────────────────────────────────
BG          = (18,  20,  24)
BOARD_BG    = (28,  30,  34)
GRID_COL    = (55,  58,  65)
SNAKE_H     = (70, 210, 110)
SNAKE_B     = (50, 180,  90)
FOOD_COL    = (225, 80,  80)
PANEL_BG    = (30,  33,  38)
NODE_COL    = (235, 235, 240)
LINE_POS    = (  0, 200,   0)
LINE_NEG    = (200,   0,   0)
TXT         = (230, 230, 235)
SEP_COL     = (80,  80,  85)
COL_LINE    = (65,  65,  70)
BTN_BG      = (70,  75,  80)
QUIT_BG     = (155, 50,  50)

UP, DOWN, LEFT, RIGHT = (0,-1), (0,1), (-1,0), (1,0)
DIRS = [UP, DOWN, LEFT, RIGHT]

# ── simple snake / food classes
class Snake:
    def __init__(self): self.reset()
    def reset(self):
        self.body = [(GRID_W//2, GRID_H//2)]
        self.dir  = random.choice(DIRS)
        self.extend = False
    def head(self): return self.body[0]
    def change(self, d):
        if d[0]==-self.dir[0] and d[1]==-self.dir[1]: return
        self.dir = d
    def step(self, wrap=True):
        hx, hy = self.head(); dx, dy = self.dir
        nx, ny = hx+dx, hy+dy
        if wrap:
            nx, ny = nx % GRID_W, ny % GRID_H
        elif not (0<=nx<GRID_W and 0<=ny<GRID_H):
            return False
        if (nx, ny) in self.body: return False
        self.body.insert(0, (nx, ny))
        if not self.extend: self.body.pop()
        self.extend = False
        return True
    def eat(self): self.extend = True

class Food:
    def __init__(self, snake): self.respawn(snake)
    def respawn(self, snake):
        while True:
            p = (random.randrange(GRID_W), random.randrange(GRID_H))
            if p not in snake.body: self.pos = p; return

# ── main game
class Game:
    FPS_MAX = 600

    def __init__(self, width=1400, height=900):
        pygame.init(); pygame.font.init()
        self.scr = pygame.display.set_mode(
            (width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption("AI Plays Simple Games")
        pygame.key.set_repeat(250, 40)

        self.ckpt = os.path.join(os.getcwd(), "checkpoints", "DQN.pt")
        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)

        self._init_fonts()
        self._calc_layout(*self.scr.get_size())

        self.clock   = pygame.time.Clock()
        self.fps_cap = FPS_INIT
        self.running = False

        self.agent = DQNAgent()
        if os.path.isfile(self.ckpt):
            self.agent.load(self.ckpt); print("[✓] checkpoint loaded")

        self.reset(full_reset=True)

    # ── layout helpers ─────────────────────────────
    def _init_fonts(self):
        h = self.scr.get_height()
        self.title_f = pygame.font.Font(None, int(0.06*h))
        self.font     = pygame.font.Font(None, int(0.03*h))
        self.small_f  = pygame.font.Font(None, int(0.024*h))

    def _calc_layout(self, w, h):
        self.W, self.H = w, h
        margin = int(0.04*w)
        top    = margin
        bar_h  = int(0.09*h)
        self.BAR_Y, self.BAR_H = h-bar_h, bar_h

        # board
        avail_h = self.BAR_Y - top
        self.GRID_SIZE = max(avail_h // (GRID_H+2), 10)
        self.BOARD_X, self.BOARD_Y = margin, top
        self.BOARD_W, self.BOARD_H = GRID_W*self.GRID_SIZE, GRID_H*self.GRID_SIZE

        # side panel
        gap = int(0.03*w)
        self.PANEL_X = self.BOARD_X + self.BOARD_W + gap
        self.PANEL_Y = self.BOARD_Y
        self.PANEL_W = w - self.PANEL_X - margin
        self.PANEL_H = self.BOARD_H

        # NN diagram area (top 45 % of panel)
        self.diag_top = self.PANEL_Y + int(self.PANEL_H*0.05)
        self.diag_h   = int(self.PANEL_H*0.45)

        col_frac = [0.25, 0.55, 0.85]
        self.IN_COL_X  = self.PANEL_X + int(self.PANEL_W*col_frac[0])
        self.HID_COL_X = self.PANEL_X + int(self.PANEL_W*col_frac[1])
        self.OUT_COL_X = self.PANEL_X + int(self.PANEL_W*col_frac[2])

        self.in_pos  = [(self.IN_COL_X,  y) for y in self._y_positions(8)]
        self.hid_pos = [(self.HID_COL_X, y) for y in self._y_positions(6)]
        self.out_pos = [(self.OUT_COL_X, y) for y in self._y_positions(4)]

        # episode log
        self.TABLE_Y = self.diag_top + self.diag_h + 12
        self.log_h   = self.PANEL_Y + self.PANEL_H - self.TABLE_Y - 6
        self.row_h   = self.small_f.get_height() + 2
        self.visible_rows = max(1, self.log_h // self.row_h)
        self.col_x = [
            self.PANEL_X + 6,
            self.PANEL_X + int(self.PANEL_W*0.22),
            self.PANEL_X + int(self.PANEL_W*0.40),
            self.PANEL_X + int(self.PANEL_W*0.58),
            self.PANEL_X + int(self.PANEL_W*0.76),
        ]

        # buttons
        btn_h = int(self.BAR_H*0.55)
        btn_w = int(0.08*w)
        gap_b = int(0.012*w)
        y_btn = self.BAR_Y + (self.BAR_H - btn_h)//2
        self.start_rect = pygame.Rect(w-margin-btn_w, y_btn, btn_w, btn_h)
        self.reset_rect = pygame.Rect(self.start_rect.left-gap_b-btn_w, y_btn, btn_w, btn_h)
        self.quit_rect  = pygame.Rect(self.reset_rect.left-gap_b-btn_w, y_btn, btn_w, btn_h)

    def _y_positions(self, n):
        step = self.diag_h / (n+1)
        return [self.diag_top + step*(i+1) for i in range(n)]

    # ── state
    def reset(self, *, full_reset: bool):
        if full_reset:
            self.log = []
            self.scroll_offset = 0
        self.snake   = Snake()
        self.food    = Food(self.snake)
        self.steps   = 0
        self.score   = 0
        self.start_t = time.perf_counter()
        self.action_time = 0.0
        self.last_action = None
        self.last_state  = [0.0]*8

    def _log_episode(self, why):
        dur = time.perf_counter() - self.start_t
        self.log.append((len(self.log)+1, dur, self.steps, self.score, why))
        if len(self.log) > 250: self.log.pop(0)

    # ── drawing: NN diagram ───────────────────────
    def _draw_nn_lines(self):
        net = self.agent.online_net
        w1 = net.fc1.weight.detach().cpu().numpy()   # (64,8)
        w2 = net.fc2.weight.detach().cpu().numpy()   # (64,64)

        hid_idx = np.linspace(0, 63, 6, dtype=int)
        w1_s = w1[hid_idx]        # (6,8)
        w2_s = w2[:4, hid_idx]    # (4,6)

        for i,(ix,iy) in enumerate(self.in_pos):
            for j,(hx,hy) in enumerate(self.hid_pos):
                wt = w1_s[j,i]; col = LINE_POS if wt>=0 else LINE_NEG
                pygame.draw.line(self.scr, col, (ix,iy), (hx,hy),
                                 max(1, int(abs(wt)*3)))
        for j,(hx,hy) in enumerate(self.hid_pos):
            for k,(ox,oy) in enumerate(self.out_pos):
                wt = w2_s[k,j]; col = LINE_POS if wt>=0 else LINE_NEG
                pygame.draw.line(self.scr, col, (hx,hy), (ox,oy),
                                 max(1, int(abs(wt)*3)))

    def _draw_nodes(self):
        r_in  = max(4, self.GRID_SIZE//5)
        r_hid = max(6, self.GRID_SIZE//4)
        r_out = max(7, self.GRID_SIZE//3)
        lbl_in  = ["Danger-F","Danger-L","Danger-R",
                   "Food-L","Food-R","Food-U","Food-D","Bias"]
        lbl_out = ["Up","Down","Left","Right"]
        blink = (time.perf_counter() - self.action_time) < BLINK_DUR

        for i,(x,y) in enumerate(self.in_pos):
            col = LINE_POS if self.last_state[i]>0.5 else NODE_COL
            pygame.draw.circle(self.scr, col, (x,y), r_in)
            self.scr.blit(self.small_f.render(lbl_in[i], True, TXT),
                          self.small_f.render(lbl_in[i], True, TXT)
                          .get_rect(midright=(x-r_in-4, y)))

        for (x,y) in self.hid_pos:
            pygame.draw.circle(self.scr, (0,120,255) if blink else NODE_COL,
                               (x,y), r_hid)

        for i,(x,y) in enumerate(self.out_pos):
            sel = blink and i==self.last_action
            pygame.draw.circle(self.scr, (0,120,255) if sel else NODE_COL,
                               (x,y), r_out)
            self.scr.blit(self.small_f.render(lbl_out[i], True, TXT),
                          self.small_f.render(lbl_out[i], True, TXT)
                          .get_rect(midleft=(x+r_out+4, y)))

    # ── drawing helpers ───────────────────────────
    def _draw_board(self):
        g = self.GRID_SIZE
        pygame.draw.rect(self.scr, BOARD_BG,
                         (self.BOARD_X, self.BOARD_Y, self.BOARD_W, self.BOARD_H))
        for x in range(1, GRID_W):
            pygame.draw.line(self.scr, GRID_COL,
                             (self.BOARD_X+x*g, self.BOARD_Y),
                             (self.BOARD_X+x*g, self.BOARD_Y+self.BOARD_H))
        for y in range(1, GRID_H):
            pygame.draw.line(self.scr, GRID_COL,
                             (self.BOARD_X, self.BOARD_Y+y*g),
                             (self.BOARD_X+self.BOARD_W, self.BOARD_Y+y*g))
        for i,(cx,cy) in enumerate(self.snake.body):
            col = SNAKE_H if i==0 else SNAKE_B
            pygame.draw.rect(self.scr, col,
                             (self.BOARD_X+cx*g+2, self.BOARD_Y+cy*g+2, g-4, g-4))
        fx,fy = self.food.pos
        pygame.draw.rect(self.scr, FOOD_COL,
                         (self.BOARD_X+fx*g+4, self.BOARD_Y+fy*g+4, g-8, g-8))

    def _draw_panel(self):
        # ── 1. panel background ──────────────────────────────────────────
        pygame.draw.rect(self.scr, PANEL_BG,
                         (self.PANEL_X, self.PANEL_Y,
                          self.PANEL_W, self.PANEL_H))

        # ── 2. neural-net diagram + captions ─────────────────────────────
        self._draw_nn_lines()
        self._draw_nodes()

        cap_surf = self.small_f.render(f"FPS {self.fps_cap}", True, TXT)
        self.scr.blit(cap_surf,
                      (self.PANEL_X + 6,
                       self.diag_top + self.diag_h - cap_surf.get_height() - 4))

        eps_val = getattr(self.agent, "eps", None)
        if eps_val is not None:
            eps_surf = self.small_f.render(f"ε {eps_val:.2f}", True, TXT)
            self.scr.blit(eps_surf,
                          (self.PANEL_X + self.PANEL_W - eps_surf.get_width() - 6,
                           self.diag_top + self.diag_h - eps_surf.get_height() - 4))

        # ── 3. episode log - header & grid lines ─────────────────────────
        heads = ["Episode", "Time", "Steps", "Score", "ROD"]
        for i, h in enumerate(heads):
            self.scr.blit(self.small_f.render(h, True, TXT),
                          (self.col_x[i], self.TABLE_Y))

        header_bottom = self.TABLE_Y + self.row_h
        pygame.draw.line(self.scr, SEP_COL,
                         (self.PANEL_X + 4, header_bottom - 1),
                         (self.PANEL_X + self.PANEL_W - 4, header_bottom - 1))

        # vertical column lines (stop at panel bottom, like original)
        col_line_y1 = self.PANEL_Y + self.PANEL_H - 4
        for x in self.col_x[1:]:
            pygame.draw.line(self.scr, COL_LINE,
                             (x - 6, header_bottom),
                             (x - 6, col_line_y1))

        # ── 4. rows (apply scroll_offset) ────────────────────────────────
        max_off = max(0, len(self.log) - self.visible_rows)
        self.scroll_offset = min(self.scroll_offset, max_off)
        start = max(0, len(self.log) - self.visible_rows - self.scroll_offset)

        y = header_bottom
        for ep, sec, stp, sc, why in self.log[start:start + self.visible_rows]:
            vals = [ep, f"{sec:.1f}", stp, sc, why]
            for i, v in enumerate(vals):
                self.scr.blit(self.small_f.render(str(v), True, TXT),
                              (self.col_x[i], y))
            # horizontal rule under the row
            pygame.draw.line(self.scr, COL_LINE,
                             (self.PANEL_X + 4, y + self.row_h - 1),
                             (self.PANEL_X + self.PANEL_W - 4, y + self.row_h - 1))
            y += self.row_h

    def _draw_buttons(self):
        for r,lbl,col in [
            (self.start_rect,"Pause" if self.running else "Start",BTN_BG),
            (self.reset_rect,"Reset",BTN_BG),
            (self.quit_rect ,"Quit", QUIT_BG),
        ]:
            pygame.draw.rect(self.scr, col, r, border_radius=6)
            surf = self.font.render(lbl, True, TXT)
            self.scr.blit(surf, surf.get_rect(center=r.center))

    def _draw(self):
        self.scr.fill(BG)
        self._draw_board()
        self._draw_panel()
        self._draw_buttons()
        self.scr.blit(self.title_f.render("Snake-AI by Prakrit", True, TXT),(self.BOARD_X, 8))
        pygame.display.flip()

    # ── main loop ─────────────────────────────────
    def run(self):
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if e.type == pygame.VIDEORESIZE:
                    self.scr = pygame.display.set_mode(
                        e.size, pygame.RESIZABLE | pygame.DOUBLEBUF)
                    self._init_fonts(); self._calc_layout(*e.size)
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_e, pygame.K_PAGEUP, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.fps_cap = min(Game.FPS_MAX, self.fps_cap+10)
                    if e.key in (pygame.K_q, pygame.K_PAGEDOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.fps_cap = max(FPS_MIN, self.fps_cap-10)
                if e.type == pygame.MOUSEBUTTONDOWN:
                    # scroll inside table
                    if e.button in (4,5):
                        mx,my = e.pos
                        x0 = self.PANEL_X+4; x1 = self.PANEL_X+self.PANEL_W-4
                        y0 = self.TABLE_Y + self.row_h; y1 = self.TABLE_Y + self.log_h
                        if x0<=mx<=x1 and y0<=my<=y1:
                            max_off = max(0, len(self.log)-self.visible_rows)
                            if e.button==4:
                                self.scroll_offset = max(0, self.scroll_offset-1)
                            else:
                                self.scroll_offset = min(max_off, self.scroll_offset+1)
                    # buttons
                    if e.button == 1:
                        if self.start_rect.collidepoint(e.pos): self.running ^= True
                        elif self.reset_rect.collidepoint(e.pos):
                            self.running = False
                            self.reset(full_reset=True)
                        elif self.quit_rect.collidepoint(e.pos):
                            pygame.quit(); sys.exit()

            if self.running:
                s = self.agent.state(self.snake, self.food)
                self.last_state = s.tolist()
                a = self.agent.act(s)
                self.last_action = a; self.action_time = time.perf_counter()
                self.snake.change(DIRS[a])

                alive = self.snake.step(wrap=True)
                rew, done, why = 0.1, False, ""
                if not alive:
                    rew, done, why = -1.0, True, "Self"
                elif self.snake.head() == self.food.pos:
                    self.snake.eat(); self.food.respawn(self.snake)
                    self.score += 1; rew = 1.0

                ns = self.agent.state(self.snake, self.food)
                self.agent.step(s,a,rew,ns,done)
                if hasattr(self.agent,"decay_eps"): self.agent.decay_eps()
                self.steps += 1

                if done:
                    self._log_episode(why)
                    self.agent.save(self.ckpt)
                    self.reset(full_reset=False)

            self._draw()
            self.clock.tick(self.fps_cap)

# ── run ───────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()
    Game(args.width, args.height).run()