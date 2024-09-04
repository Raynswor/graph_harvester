import pygame

from geometry_objects import *
from typing import List

SCALING_FACTOR = 1

OFFSET_X = 000
OFFSET_Y = 000


def scale(p):
    return ((p[0] - OFFSET_X) * SCALING_FACTOR, (p[1] - OFFSET_Y) * SCALING_FACTOR)


class Drawer:
    def __init__(self) -> None:
        pygame.init()
        infoObject = pygame.display.Info()
        self.screen = pygame.display.set_mode(
            (infoObject.current_w - 100, infoObject.current_h - 100)
        )

        self.my_font = pygame.font.SysFont("Comic Sans MS", 15)

        self.clock = pygame.time.Clock()

        self.colors = [
            pygame.Color("green"),
            pygame.Color("yellow"),
            pygame.Color("cyan"),
            pygame.Color("magenta"),
            pygame.Color("white"),
        ]

    def check_for_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                running = False

    def draw_objects(
        self,
        circles: List[Circle],
        lines: List[Line],
        rects: List[Rect],
        quads: List[Rect],
        beziers: List[Bezier],
        labels: List[Label],
    ):
        self.screen.fill(pygame.Color("gray"))

        j = 0
        for rect in rects + quads:
            pygame.draw.rect(
                self.screen,
                self.colors[j % len(self.colors)],
                pygame.Rect(
                    scale(rect.topLeft),
                    (rect.width * SCALING_FACTOR, rect.height * SCALING_FACTOR),
                ),
                2,
            )
            surf = self.my_font.render(
                str(len(list(filter(lambda circle: circle.draw, circles))) + j),
                True,
                pygame.Color("black"),
            )
            self.screen.blit(surf, scale(rect.topLeft))
            j += 1

        j = 0
        for circle in circles:
            if not circle.draw:
                continue
            pygame.draw.circle(
                self.screen,
                # pygame.Color("red"),
                self.colors[j % len(self.colors)],
                scale(circle.center),
                circle.radius * SCALING_FACTOR,
                # CONNECT_LINES_THRESHOLD,
            )
            surf = self.my_font.render(str(j), True, pygame.Color("black"))
            self.screen.blit(surf, scale(circle.center))
            j += 1

        j = 0
        for line in lines:
            pygame.draw.line(
                self.screen,
                pygame.Color("blue"),
                # self.colors[j % len(self.colors)],
                scale(line.start),
                scale(line.stop),
                2,
            )
            surf = self.my_font.render(str(j), True, pygame.Color("black"))
            self.screen.blit(surf, scale(line.start))
            j += 1

        for j in range(len(beziers)):
            bezier = beziers[j]

            scaled_points = [scale(p) for p in bezier.points]
            color = self.colors[j % len(self.colors)]
            pygame.draw.lines(
                self.screen,
                color,
                False,
                scaled_points,
                2,
            )
            # surf = self.my_font.render(str(j), True, pygame.Color("black"))
            # self.screen.blit(surf, scale(bezier.start))
            # pygame.draw.circle(
            #     self.screen,
            #     pygame.Color("red"),
            #     scale(bezier.start),
            #     4,
            #     # CONNECT_LINES_THRESHOLD,
            # )

        for label in labels:

            surf = self.my_font.render(label.content, True, pygame.Color("black"))
            self.screen.blit(surf, scale((label.topLeft[0], label.topLeft[1])))

            # pygame.draw.rect(
            #     self.screen,
            #     pygame.Color("red"),
            #     Rect(
            #         scale((label.topLeft[0], label.topLeft[1])),
            #         scale(
            #             (label.width, label.height),
            #         ),
            #     ),
            #     2,
            # )

    def update(self):
        pygame.display.flip()
        self.clock.tick(10)
