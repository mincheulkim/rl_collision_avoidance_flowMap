'''
import math 
import numpy as np 
import matplotlib.pyplot as plt 
EXTEND_AREA = 10.0   # 확장영역
show_animation = True

def calc_grid_map_config(ox, oy, xyreso): 
    minx = round(min(ox) - EXTEND_AREA / 2.0) 
    miny = round(min(oy) - EXTEND_AREA / 2.0) 
    maxx = round(max(ox) + EXTEND_AREA / 2.0) 
    maxy = round(max(oy) + EXTEND_AREA / 2.0) 
    xw = int(round((maxx - minx) / xyreso)) 
    yw = int(round((maxy - miny) / xyreso)) 
    return minx, miny, maxx, maxy, xw, yw

class precastDB:              # 해당 셀의 위치 px,py와 원점으로부터의 거리 d, 각 angle, 그리고 격자 인덱스 ix, iy를 가지는 클래스
    def __init__(self): 
        self.px = 0.0 
        self.py = 0.0 
        self.d = 0.0 
        self.angle = 0.0 
        self.ix = 0 
        self.iy = 0 
        
    def __str__(self): 
        return str(self.px) + "," + str(self.py) + "," + str(self.d) + "," + str(self.angle) 
    
def atan_zero_to_twopi(y, x): 
    angle = math.atan2(y, x) 
    if angle < 0.0: 
        angle += math.pi * 2.0 
        
    return angle


def precasting(minx, miny, xw, yw, xyreso, yawreso): 
    precast = [[] for i in range(int(round((math.pi * 2.0) / yawreso)) + 1)]    # 2*pi/각도해상도 = 6.28/0.25=24.96 반올림후 +1하면 26번의 리스트
    # precast.shape = (1, 26)
    
    for ix in range(xw):      # 모든 격자에 대해 반복
        for iy in range(yw): 
            px = ix * xyreso + minx 
            py = iy * xyreso + miny        # 첫번째 격자의 x,y좌표를 구함 
            
            d = math.hypot(px, py)         # 원점에서 첫 격자까지의 거리d
            angle = atan_zero_to_twopi(py, px)           # 원점에서 첫 격자까지 각도 angel
            angleid = int(math.floor(angle / yawreso))   # 각도를 해상도만큼 나누어 만든 정수 angleid
            
            pc = precastDB()               # 첫번째 셀에 대한 정보를 담을 전투사 데이터베이스 precastDB()
            
            pc.px = px                         # 이 전투사 DB에다가 첫번째 셀의 위치와 거리, 각도, 인덱스 등을 담음
            pc.py = py 
            pc.d = d 
            pc.ix = ix 
            pc.iy = iy 
            pc.angle = angle 
            
            precast[angleid].append(pc)    # 아까 텅빈 리스트들로 이루어진 (1,26)크기의 2차원 행렬에 해당 angle id 번호에 전투사 데이터베이스를 추가
            # 모든 셀에 대해 전투사데이터베이스 만들고 난 후
            
    return precast   # 전체 행렬을 반환

def generate_ray_casting_grid_map(ox, oy, xyreso, yawreso): 
    minx, miny, maxx, maxy, xw, yw = calc_grid_map_config(ox, oy, xyreso) 
    
    
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]    # 2차원 격자지도 pmap
    
    precast = precasting(minx, miny, xw, yw, xyreso, yawreso)   # 전투사 precasting
    
    for (x, y) in zip(ox, oy):          # 각 랜드마크의 좌표, 첫번째 -> 두번째 랜드마크 순으로 반복 수행
        d = math.hypot(x, y)            # 랜드마크와의 거리
        angle = atan_zero_to_twopi(y, x)    # 각
        angleid = int(math.floor(angle / yawreso))    # 각 아이디를 계산
        
        gridlist = precast[angleid]               # 해당 랜드마크의 angleid와 동일한 격자 셀들의 목록 gridlist를  가져옴
        ix = int(round((x - minx) / xyreso))      # 첫번째 랜드마크의 격자 인덱스를 계산
        iy = int(round((y - miny) / xyreso)) 
        
        for grid in gridlist:                   # 격자 목록에 존재하는 각 격자들을 돌리면서, 현재 랜드마크의 거리보다 격자의 거리가 먼지 비교
            if grid.d > d:                       
                pmap[grid.ix][grid.iy] = 0.5    # 현재 랜드마크의 거리보다 격자가 뒤에 있으면 알수 없음이므로 0.5
                
        pmap[ix][iy] = 1.0                      # 현재 격자는 알고 있으므로 1
    return pmap, minx, maxx, miny, maxy, xyreso

def draw_heatmap(data, minx, maxx, miny, maxy, xyreso): 
    x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso), slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)] 
    plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues) 
    plt.axis("equal")

def main(): 
    print(__file__ + " start!!") 
    xyreso = 0.1 # x-y grid resolution [m]   0.25           x,y grid resolution
    yawreso = np.deg2rad(3.14) # yaw angle resolution [rad]   10.0    # 회전각 yaw 해상도  0.00613592315153
    
    for i in range(1): 
        ox = (np.random.rand(4) - 0.5) * 10.0 
        oy = (np.random.rand(4) - 0.5) * 10.0 
        pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(
            ox, oy, xyreso, yawreso) 
        
        if show_animation: # pragma: no cover 
            plt.cla() 
            # for stopping simulation with the esc key. 
            plt.gcf().canvas.mpl_connect('key_release_event', 
                                         lambda event: [exit(0) if event.key == 'escape' else None]) 
            draw_heatmap(pmap, minx, maxx, miny, maxy, xyreso) 
            plt.plot(ox, oy, "xr") 
            plt.plot(0.0, 0.0, "ob") 
            plt.pause(1.0) 
            
if __name__ == '__main__': 
    main()

'''

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

EXTEND=AREA