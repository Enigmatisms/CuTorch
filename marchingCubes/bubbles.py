import torch
import march
import cv2 as cv
import numpy as np
import profile

bubble_num = 5
size = torch.LongTensor([800, 600])
velos = torch.zeros(bubble_num, 2)

def generateBubbles(number = 5):
    i = 0
    bubbles = []
    while i < number:
        center = size.type(torch.FloatTensor) * (0.6 * torch.rand(1) + 0.2)
        to_zero = center.clone()
        cand = torch.cat([size.type(torch.FloatTensor) - center, to_zero])
        max_radius = 0.5 * min(cand)
        if max_radius <= 10.0:
            continue
        radius = (torch.rand(1) * 0.75 + 0.25) * max_radius
        if radius <= 6.0:
            radius = (torch.rand(1) * 0.75 + 0.25) * max_radius
        i += 1
        x, y = center
        bubbles.append([x, y, radius])
    return torch.FloatTensor(bubbles)

def randomInitialVelocity(number = 5):
    vels = []
    for i in range(number):
        vel = torch.normal(0, 2, (1, 2))
        while vel.norm() < 1.0:
            vel = torch.normal(0, 2, (1, 2))
        vels.append(vel)
    return torch.cat(vels, dim = 0)

def bouncingWithNoise(bubbles:torch.Tensor, number = 5):
    for i in range(number):
        x, y, radius = bubbles[i]
        vx, vy = velos[i]
        if x + radius + vx > size[0] or x - radius + vx < 0:
            velos[i, 0] = -velos[i, 0] + torch.normal(0, 0.1, (1, 1)).view(1)
        if y + radius + vy > size[1] or y - radius + vy < 0:
            velos[i, 1] = -velos[i, 1] + torch.normal(0, 0.1, (1, 1)).view(1)

def drawContours(ct_raw:torch.Tensor):
    w, h = size
    img = np.zeros([h, w, 3], dtype = np.uint8)
    for vec in ct_raw:
        y1, x1, y2, x2, x0, y0 = vec
        img = cv.line(img, (1 - x1 + x0, y1 + y0), (1 - x2 + x0, y2 + y0), (0, 255, 0), 1)
    return img

def bubbleMove(bubbles:torch.Tensor):
    for i, bb in enumerate(bubbles):
        res = bb[:2] + velos[i]
        bb[0] = max(min(size[0] - 1e-3, res[0]), 1e-3) 
        bb[1] = max(min(size[1] - 1e-3, res[1]), 1e-3) 
    return bubbles

def main():
    bubbles = generateBubbles(bubble_num)
    # for i in range(30):
    while True:
        ct_raw = march.marchingSquare(bubbles / 2.0)
        ct_raw *= 2.0
        bouncingWithNoise(bubbles, bubble_num)
        bubbles = bubbleMove(bubbles)
        img = drawContours(ct_raw)
        cv.imshow('disp', img)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord(' '):
            cv.waitKey(0)


if __name__ == "__main__":
    velos = randomInitialVelocity(bubble_num)
    main()    

        