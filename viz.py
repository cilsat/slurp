import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json
import math

def getData(folder='data/Imod Jakarta/', filename='Boreholes_Jakarta.ipf'):
    ipf = open(folder+filename).read().replace('\r','').split('\n')
    ipf = [d for d in ipf if len(d.split(',')) == 7]

    data = []
    for d in ipf:
        x, y, name, _,_,_,_ = d.split(',')

        txt = open(folder+name.replace('\\', '/')+'.txt').read().replace('\r','').split('\n')
        txt = [t for t in txt if len(t.split(',')) == 2]

        # lapisan = []
        # details = {}
        # for i in range(0, len(txt)-1):
        #     z1,jenis = txt[i].split(',')
        #     z2,_ = txt[i+1].split(',')
        #     if jenis != '-9999.000' and _ != '-9999.000':
        #         lapisan.append((jenis,float(z1),float(z2)))
        #         details[jenis] = []

        # cur_jenis, cur_z1, cur_z2 = None, None, None
        # for jenis, z1, z2 in lapisan:
        #     if jenis == cur_jenis:
        #         cur_z2 = z2
        #     else:
        #         if cur_jenis != None:
        #             details[cur_jenis].append((float(cur_z1), float(cur_z2)))
        #         cur_jenis, cur_z1, cur_z2 = jenis, z1, z2
        # if cur_jenis != None:
        #     details[cur_jenis].append((float(cur_z1), float(cur_z2)))

        details = {}
        for i in range(0, len(txt)-1):
            z1,jenis = txt[i].split(',')
            z2,_ = txt[i+1].split(',')
            if jenis != '-9999.000' and _ != '-9999.000':
                if jenis not in details:
                    details[jenis] = []
                details[jenis].append((float(z1),float(z2)))

        data.append({
            'x': float(x),
            'y': float(y),
            'details': details,
        })

    return data

def plot(data):
    ax = plt.gca(projection='3d')
    dots = {
        'sand': {'x':[], 'y':[], 'z':[]},
        'nonsand': {'x':[], 'y':[], 'z':[]},
    }
    for datum in data:
        x, y, details = datum['x'], datum['y'], datum['details']
        for jenis in details:
            liths = details[jenis]
            dot = dots['sand'] if jenis == 'sand' else dots['nonsand']
            for z1, z2 in liths:
                mini, maxi = min(z1, z2), max(z1, z2)
                while mini < maxi:
                    dot['x'].append(x)
                    dot['y'].append(y)
                    dot['z'].append(mini)
                    mini += 1
                dot['x'].append(x)
                dot['y'].append(y)
                dot['z'].append(maxi)
    ax.scatter(dots['sand']['x'], dots['sand']['y'], dots['sand']['z'], c='yellow', marker='|', depthshade=False)
    ax.scatter(dots['nonsand']['x'], dots['nonsand']['y'], dots['nonsand']['z'], c='green', s=1, marker='|', depthshade=False)
    plt.show()
