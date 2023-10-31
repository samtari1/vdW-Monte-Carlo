# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 07:26:41 2023

@author: nvnca
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:50:33 2023

@author: samtari
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import random
import pandas as pd
import os
import re
import pickle
import warnings
from tqdm import trange
from itertools import zip_longest

from functools import partial
import multiprocessing
from multiprocessing import Pool
################################################################################################
######################## AA/CG LJ energy calculation functions ##########################
def scaleEpsilonForCG(AASigma,AACubeSize,AAEpsilon,CGSigma,CGCubeSize,cutoffAA, cutoffCG):
    # scale epsilon of CG model to match the effect of epsilon in AA model
    # which is to match the minimum of energy profile of CG with AA 
    EnergyListCG = []
    distanceListCG = []
    EnergyListAA = []
    distanceListAA = []
    df = pd.DataFrame()
    # scaning energy vs distance curve for two AA cubes
    startDistance = 0.01 * AASigma 
    endDistance = AASigma
    distanceInterval = 0.1*AASigma
    numDataPoints = int((endDistance-startDistance)/distanceInterval)
    print('Start scaling Epsilon for CG based on AA:')
    for i in trange(numDataPoints):
        currentDistance = startDistance + i * distanceInterval
        # AA cube 1
        COMX,COMY,COMZ = 0, 0, 0
        VectorX, VectorY, VectorZ = [1,0,0],[0,1,0],[0,0,1]
        atomList1 = findAtomPositionsInCube(AACubeSize,AASigma,COMX,COMY,COMZ,VectorX, VectorY, VectorZ)
        # AA cube 2
        CubeSideLength = AASigma * AACubeSize
        COMX,COMY,COMZ = currentDistance + CubeSideLength, 0, 0
        VectorX, VectorY, VectorZ = [1,0,0],[0,1,0],[0,0,1]
        atomList2 = findAtomPositionsInCube(AACubeSize,AASigma,COMX,COMY,COMZ,VectorX, VectorY, VectorZ)
        # energy between AA cube 1 and AA cube 2
        CubeCubeEnergy = LJPotentialBetweenTwoCubes(AASigma, AAEpsilon, cutoffAA, atomList1, atomList2)
        # print(currentDistance, CubeCubeEnergy)
        distanceListAA.append(currentDistance)
        EnergyListAA.append(CubeCubeEnergy)    
    AAminEnergy = min(EnergyListAA) # find the minimum energy of AA   
    
    # scaning energy vs distance curve for two CG cubes
    startDistance = 0.01 * AASigma 
    endDistance = 3 * AASigma
    distanceInterval = 0.1*AASigma
    numDataPoints = int((endDistance-startDistance)/distanceInterval)
    for i in trange(numDataPoints):
        currentDistance = startDistance + i * distanceInterval
        # CG cube 1
        COMX,COMY,COMZ = 0, 0, 0
        VectorX, VectorY, VectorZ = [1,0,0],[0,1,0],[0,0,1]
        atomList1 = findAtomPositionsInCube(CGCubeSize,CGSigma,COMX,COMY,COMZ,VectorX, VectorY, VectorZ)
        # CG cube 2
        COMX,COMY,COMZ = i + CubeSideLength, 0, 0
        VectorX, VectorY, VectorZ = [1,0,0],[0,1,0],[0,0,1]
        atomList2 = findAtomPositionsInCube(CGCubeSize,CGSigma,COMX,COMY,COMZ,VectorX, VectorY, VectorZ)
        # energy between CG cube 1 and CG cube 2
        CubeCubeEnergy = LJPotentialBetweenTwoCubes(CGSigma, AAEpsilon, cutoffCG, atomList1, atomList2)
        distanceListCG.append(i)
        EnergyListCG.append(CubeCubeEnergy)  
    CGminEnergy = min(EnergyListCG) # find the minimum energy of CG
    
    # calculate scale factor
    ScaleFactor = AAminEnergy / CGminEnergy
    ScaleFactorList = [ScaleFactor]
    print('ScaleFactor = ', ScaleFactor)
    # scale CG energy curve to make its minimum equal to the minimum of AA energy curve
    ScaledCGEnergyList = [i*ScaleFactor for i in EnergyListCG]
    
    # Use zip_longest to align lists and insert None for missing values
    aligned_lists = list(zip_longest(distanceListAA, EnergyListAA, distanceListCG, EnergyListCG, ScaledCGEnergyList, ScaleFactorList, fillvalue=None))

    # Create a DataFrame from the aligned lists
    data = {'AA_Distance':      [item[0] for item in aligned_lists],
            'AA_Energy':        [item[1] for item in aligned_lists],
            'CG_Distance':      [item[2] for item in aligned_lists],
            'CG_Energy':        [item[3] for item in aligned_lists],
            'Scaled_CG_Energy': [item[4] for item in aligned_lists],
            'Scale_Factor':     [item[5] for item in aligned_lists]
            
            }
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv('Scaling CG Energy based on AA Energy.csv', index=False)
    
    # Plotting the curves
    plt.plot(distanceListAA, EnergyListAA, label='AA')
    plt.plot(distanceListCG, EnergyListCG, label='CG')    
    plt.plot(distanceListCG, ScaledCGEnergyList, label='Scaled CG')    
    plt.legend()
    plt.xlabel('Distance (Å)')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Plot Energy Curves')
    plt.savefig('Scaling Energy.png')
    plt.show()
    
    return ScaleFactor

def LJPotentialBetweenTwoAtoms(Sigma, Epsilon, distance):
    SigmaDividedByDistance = Sigma/distance
    Energy = 4 * Epsilon * (SigmaDividedByDistance**12 - SigmaDividedByDistance**6)
    return Energy

def LJPotentialBetweenTwoCubes(Sigma, Epsilon, LJCutOff, atomList1, atomList2):
    CubeCubeEnergy = 0
    for atom1 in atomList1:
        for atom2 in atomList2:
            Atom1X = atom1[0]
            Atom1Y = atom1[1]
            Atom1Z = atom1[2]
            Atom2X = atom2[0]
            Atom2Y = atom2[1]
            Atom2Z = atom2[2]
            distance = ((Atom1X-Atom2X)**2 + (Atom1Y-Atom2Y)**2 + (Atom1Z-Atom2Z)**2)**(1/2)
            if distance <= LJCutOff:
                energy = LJPotentialBetweenTwoAtoms(Sigma, Epsilon, distance)
                CubeCubeEnergy += energy
                # print('yes')
    return CubeCubeEnergy
######################## AA/CG LJ energy calculation functions ##########################
################################################################################################


################################################################################################
######################## vdW energy calculation functions ##########################
def ecalcgen(param, cene, n, c3now, caseind):
    a1, a2, b1, b2, c1, c2, c3, ym, yp, AtomSigma = param
    if caseind == 1:
        u1 = (yp - ym) * (a1 * c2 + c3now) ** (1 - n)
        u2 = (ym - yp) * (a2 * c2 + c3now) ** (1 - n)
        energy = cene * (u1 + u2) * AtomSigma ** (n - 2) / (c2 * (n - 1) * (n - 2))
    elif caseind == 2:
        u1 = (yp - ym) * (a1 * c2 + c3now) ** (1 - n)
        u2 = -(a2 * c2 + c3now + (c1 + b2 * c2) * ym) ** (2 - n) + (a2 * c2 + c3now + (c1 + b2 * c2) * yp) ** (2 - n)
        u2 = u2 / ((c1 + b2 * c2))
        energy = cene * (u1 + u2) * AtomSigma ** (n - 2) / (c2 * (n - 1) * (n - 2))
    elif caseind == 3:
        u1 = (a1 * c2 + c3now + (c1 + b1 * c2) * ym) ** (2 - n) - (a1 * c2 + c3now + (c1 + b1 * c2) * yp) ** (2 - n)
        u1 = u1 / ((c1 + b1 * c2))
        u2 = (ym - yp) * (a2 * c2 + c3now) ** (1 - n)
        energy = cene * (u1 + u2) * AtomSigma ** (n - 2) / (c2 * (n - 1) * (n - 2))
    elif caseind == 4:
        u1 = (a1 * c2 + c3now + (c1 + b1 * c2) * ym) ** (2 - n) - (a1 * c2 + c3now + (c1 + b1 * c2) * yp) ** (2 - n)
        u1 = u1 / ((c1 + b1 * c2))
        u2 = -(a2 * c2 + c3now + (c1 + b2 * c2) * ym) ** (2 - n) + (a2 * c2 + c3now + (c1 + b2 * c2) * yp) ** (2 - n)
        u2 = u2 / ((c1 + b2 * c2))
        energy = cene * (u1 + u2) * AtomSigma ** (n - 2) / (c2 * (n - 1) * (n - 2))
    return energy

def ecalcc2(param, cene, n, c3now, caseind):
    a1, a2, b1, b2, c1, c2, c3, ym, yp, AtomSigma = param
    u1 = (c1 * yp + c3now) ** (1 - n) * ((a1 * c1 - a2 * c1) * (n - 2) + (b1 - b2) * (c3now + c1 * (n - 1) * yp))
    u2 = -(c1 * ym + c3now) ** (1 - n) * ((a1 * c1 - a2 * c1) * (n - 2) + (b1 - b2) * (c3now + c1 * (n - 1) * ym))
    energy = cene * AtomSigma ** (n - 2) / ((c1) ** 2) / (n - 2) / (n - 1) * (u1 + u2)
    return energy

def ecalcc1(param, cene, n, c3now, caseind):
    a1, a2, b1, b2, c1, c2, c3, ym, yp, AtomSigma = param
    if caseind == 1:
        energy = cene * AtomSigma ** (n - 2) / c2 / (1 - n) * ((yp - ym) * (a2 * c2 + c3now) ** (1 - n) + (ym - yp) * (a1 * c2 + c3now) ** (1 - n))
    elif caseind == 2:
        u1 = (yp - ym) * (a2 * c2 + c3now) ** (1 - n)
        u2 = (a1 * c2 + c3now + b1 * c2 * ym) ** (2 - n) - (a1 * c2 + c3now + b1 * c2 * yp) ** (2 - n)
        u2 = u2 / (b1 * c2 * (2 - n))
        energy = cene * AtomSigma ** (n - 2) / c2 / (1 - n) * (u1 + u2)
    elif caseind == 3:
        u1 = (ym - yp) * (a1 * c2 + c3now) ** (1 - n)
        u2 = (a2 * c2 + c3now + b2 * c2 * yp) ** (2 - n) - (a2 * c2 + c3now + b2 * c2 * ym) ** (2 - n)
        u2 = u2 / (b2 * c2 * (2 - n))
        energy = cene * AtomSigma ** (n - 2) / c2 / (1 - n) * (u1 + u2)
    elif caseind == 4:
        u1 = -(a2 * c2 + c3now + b2 * c2 * ym) ** (2 - n) + (a2 * c2 + c3now + b2 * c2 * yp) ** (2 - n)
        u1 = u1 / (2 * b2 * c2 - b2 * c2 * n)
        u2 = (a1 * c2 + c3now + b1 * c2 * ym) ** (2 - n) - (a1 * c2 + c3now + b1 * c2 * yp) ** (2 - n)
        u2 = u2 / (2 * b1 * c2 - b1 * c2 * n)
        energy = cene * (u1 + u2) * AtomSigma ** (n - 2) / (c2 * (1 - n))
    return energy

def calc_gen(region, facein, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon):
    dslim = 2.0
    slopetol = 1e-5
    temper = np.array(facein)[np.argsort(np.array(facein)[:, 1])]
    face = np.zeros((4, 3))
    face[0, :] = temper[0, :]
    face[3, :] = temper[3, :]

    if temper[1][2] > temper[2][2]:
        face[1, :] = temper[1, :]
        face[2, :] = temper[2, :]
    else:
        face[1, :] = temper[2, :]
        face[2, :] = temper[1, :]

    yp = region[1][0]
    ym = region[0][0]
    b2 = (region[1][1] - region[0][1]) / (region[1][0] - region[0][0])
    a2 = region[0][1] - b2 * region[0][0]
    b1 = (region[3][1] - region[2][1]) / (region[3][0] - region[2][0])
    a1 = region[2][1] - b1 * region[2][0]

    AA = (face[1][1] - face[0][1]) * (face[2][2] - face[0][2]) - (face[1][2] - face[0][2]) * (face[2][1] - face[0][1])
    BB = (face[1][2] - face[0][2]) * (face[2][0] - face[0][0]) - (face[1][0] - face[0][0]) * (face[2][2] - face[0][2])
    CC = ((face[1][0] - face[0][0]) * (face[2][1] - face[0][1])) - ((face[1][1] - face[0][1]) * (face[2][0] - face[0][0]))

    c1 = -BB / AA
    c2 = -CC / AA
    c3 = face[0][0] - lc - c1 * face[0][1] - c2 * face[0][2]
    debug = 0
    param = [a1, a2, b1, b2, c1, c2, c3, ym, yp, AtomSigma]

    if c3 > dslim:
        if abs(c2) < slopetol and abs(c1) < slopetol:
            area = ((region[0][1] - region[2][1]) + (region[1][1] - region[3][1])) * (region[1][0] - region[0][0]) / 2
            n = atre2
            uatr = catr2 * area * c3**(-n)
            urep = 0.0
        elif abs(c1) < slopetol:
            if abs(b1) < slopetol and abs(b2) < slopetol:
                caseindc1 = 1
            elif abs(b2) < slopetol:
                caseindc1 = 2
            elif abs(b1) < slopetol:
                caseindc1 = 3
            else:
                caseindc1 = 4
            uatr = ecalcc1(param, catr2, atre2, c3, caseindc1)
            urep = 0.0
        elif abs(c2) < slopetol:
            n = atre2
            uatr = ecalcc2(param, catr2, atre2, c3, 1)
            urep = 0.0
        else:
            check1 = abs(c1 + b1 * c2)
            check2 = abs(c1 + b2 * c2)
            if check1 < slopetol and check2 < slopetol:
                caseindgen = 1
            elif check1 < slopetol:
                caseindgen = 2
            elif check2 < slopetol:
                caseindgen = 3
            else:
                caseindgen = 4
            uatr = ecalcgen(param, catr2, atre2, c3, caseindgen)
            urep = 0.0
    else:
        c3t = dslim
        if abs(c2) < slopetol and abs(c1) < slopetol:
            area = ((region[0][1] - region[2][1]) + (region[1][1] - region[3][1])) * (region[1][0] - region[0][0]) / 2
            n = atre1
            uatr = catr1 * area * c3**(-n)
            nt2 = atre2
            nt1 = atre1
            uatrt2 = catr2 * area * c3t**(-nt2)
            uatrt1 = catr1 * area * c3t**(-nt1)
            uatr = uatr + uatrt2 - uatrt1
            n = repe
            urep = crep * area * c3**(-n)
            ureptemp = crep * area * c3t**(-n)
            urep = urep - ureptemp
        elif abs(c1) < slopetol:
            if abs(b1) < slopetol and abs(b2) < slopetol:
                caseindc1 = 1
            elif abs(b2) < slopetol:
                caseindc1 = 2
            elif abs(b1) < slopetol:
                caseindc1 = 3
            else:
                caseindc1 = 4
            uatr = ecalcc1(param, catr1, atre1, c3, caseindc1)
            uatrt1 = ecalcc1(param, catr1, atre1, c3t, caseindc1)
            uatrt2 = ecalcc1(param, catr2, atre2, c3t, caseindc1)
            uatr = uatr + uatrt2 - uatrt1
            urep = ecalcc1(param, crep, repe, c3, caseindc1)
            ureptemp = ecalcc1(param, crep, repe, c3t, caseindc1)
            urep = urep - ureptemp
        elif abs(c2) < slopetol:
            uatr = ecalcc2(param, catr1, atre1, c3, 1)
            uatrt1 = ecalcc2(param, catr1, atre1, c3t, 1)
            uatrt2 = ecalcc2(param, catr2, atre2, c3t, 1)
            uatr = uatr + uatrt2 - uatrt1
            urep = ecalcc2(param, crep, repe, c3, 1)
            ureptemp = ecalcc2(param, crep, repe, c3t, 1)
            urep = urep - ureptemp
        else:
            check1 = abs(c1 + b1 * c2)
            check2 = abs(c1 + b2 * c2)
            if check1 < slopetol and check2 < slopetol:
                caseindgen = 1
            elif check1 < slopetol:
                caseindgen = 2
            elif check2 < slopetol:
                caseindgen = 3
            else:
                caseindgen = 4
            uatr = ecalcgen(param, catr1, atre1, c3, caseindgen)
            uatrt1 = ecalcgen(param, catr1, atre1, c3t, caseindgen)
            uatrt2 = ecalcgen(param, catr2, atre2, c3t, caseindgen)
            uatr = uatr + uatrt2 - uatrt1
            urep = ecalcgen(param, crep, repe, c3, caseindgen)
            ureptemp = ecalcgen(param, crep, repe, c3t, caseindgen)
            urep = urep - ureptemp

    uatr = uatr * Epsilon
    urep = urep * Epsilon
    utot = uatr + urep

    return uatr, urep, utot, param

def region_calc(facein, CubeSideLength, AtomSigma, lc):
    reg_tol = lc + 1e-9
    facein = np.array(facein)
    temper = facein[np.argsort(facein[:, 1])]
    face2 = np.zeros((4, 3))
    face2[0, :] = temper[0, :]
    
    if temper[1, 2] > temper[2, 2]:
        face2[1, :] = temper[1, :]
        face2[2, :] = temper[2, :]
    else:
        face2[1, :] = temper[2, :]
        face2[2, :] = temper[1, :]
    
    face2[3, :] = temper[3, :]
    xcount = 0
    xsave = []
    
    # Line1
    L1_slope = (face2[1, 1] - face2[0, 1]) / (face2[1, 2] - face2[0, 2])
    L1y1 = (-lc - face2[1, 2]) * L1_slope + face2[1, 1]
    L1y2 = (lc - face2[1, 2]) * L1_slope + face2[1, 1]
    
    if L1y1 > face2[0, 1] and L1y1 < face2[1, 1] and L1y1 <= lc and L1y1 >= -lc:
        xcount += 1
        xsave.append(L1y1)
    
    if L1y2 > face2[0, 1] and L1y2 < face2[1, 1] and L1y2 <= lc and L1y2 >= -lc:
        xcount += 1
        xsave.append(L1y2)
    
    # Line2
    L2_slope = (face2[3, 1] - face2[1, 1]) / (face2[3, 2] - face2[1, 2])
    L2y1 = (-lc - face2[1, 2]) * L2_slope + face2[1, 1]
    L2y2 = (lc - face2[1, 2]) * L2_slope + face2[1, 1]
    
    if L2y1 < face2[3, 1] and L2y1 > face2[1, 1] and L2y1 <= lc and L2y1 >= -lc:
        xcount += 1
        xsave.append(L2y1)
    
    if L2y2 < face2[3, 1] and L2y2 > face2[1, 1] and L2y2 <= lc and L2y2 >= -lc:
        xcount += 1
        xsave.append(L2y2)
    
    # Line3
    L3_slope = (face2[2, 1] - face2[0, 1]) / (face2[2, 2] - face2[0, 2])
    L3y1 = (-lc - face2[2, 2]) * L3_slope + face2[2, 1]
    L3y2 = (lc - face2[2, 2]) * L3_slope + face2[2, 1]
    
    if L3y1 > face2[0, 1] and L3y1 < face2[2, 1] and L3y1 <= lc and L3y1 >= -lc:
        xcount += 1
        xsave.append(L3y1)
    
    if L3y2 > face2[0, 1] and L3y2 < face2[2, 1] and L3y2 <= lc and L3y2 >= -lc:
        xcount += 1
        xsave.append(L3y2)
    
    # Line4
    L4_slope = (face2[3, 1] - face2[2, 1]) / (face2[3, 2] - face2[2, 2])
    L4y1 = (-lc - face2[2, 2]) * L4_slope + face2[2, 1]
    L4y2 = (lc - face2[2, 2]) * L4_slope + face2[2, 1]
    
    if L4y1 < face2[3, 1] and L4y1 > face2[2, 1] and L4y1 <= lc and L4y1 >= -lc:
        xcount += 1
        xsave.append(L4y1)
    
    if L4y2 < face2[3, 1] and L4y2 > face2[2, 1] and L4y2 <= lc and L4y2 >= -lc:
        xcount += 1
        xsave.append(L4y2)
    
    for i in range(4):
        if face2[i, 1] >= -lc and face2[i, 1] <= lc and face2[i, 2] >= -lc and face2[i, 2] <= lc:
            xcount += 1
            xsave.append(face2[i, 1])
    
    if face2[0, 1] < -lc:
        xcount += 1
        xsave.append(-lc)
    
    if face2[3, 1] > lc:
        xcount += 1
        xsave.append(lc)
    
    if xcount > 0:
        xsort = np.unique(np.sort(xsave))
        region = np.zeros((xsort.shape[0] - 1, 4, 2))
        regioncount = 0
        
        for i1 in range(1, len(xsort)):
            RYLB = xsort[i1 - 1]
            RYUB = xsort[i1]
            
            if RYLB < face2[1, 1]:
                slope = (face2[1, 2] - face2[0, 2]) / (face2[1, 1] - face2[0, 1])
                
                if np.isinf(slope):
                    RZUB1 = face2[1, 2]
                else:
                    RZUB1 = slope * (RYLB - face2[1, 1]) + face2[1, 2]
            else:
                slope = (face2[3, 2] - face2[1, 2]) / (face2[3, 1] - face2[1, 1])
                
                if np.isinf(slope):
                    RZUB1 = face2[1, 2]
                else:
                    RZUB1 = slope * (RYLB - face2[1, 1]) + face2[1, 2]
            
            if RYUB < face2[1, 1]:
                slope = (face2[1, 2] - face2[0, 2]) / (face2[1, 1] - face2[0, 1])
                
                if np.isinf(slope):
                    RZUB2 = face2[1, 2]
                else:
                    RZUB2 = slope * (RYUB - face2[1, 1]) + face2[1, 2]
            else:
                slope = (face2[3, 2] - face2[1, 2]) / (face2[3, 1] - face2[1, 1])
                
                if np.isinf(slope):
                    RZUB2 = face2[1, 2]
                else:
                    RZUB2 = slope * (RYUB - face2[1, 1]) + face2[1, 2]
            
            if RYLB < face2[2, 1]:
                slope = (face2[2, 2] - face2[0, 2]) / (face2[2, 1] - face2[0, 1])
                
                if np.isinf(slope):
                    RZLB1 = face2[2, 2]
                else:
                    RZLB1 = slope * (RYLB - face2[2, 1]) + face2[2, 2]
            else:
                slope = (face2[3, 2] - face2[2, 2]) / (face2[3, 1] - face2[2, 1])
                
                if np.isinf(slope):
                    RZLB1 = face2[2, 2]
                else:
                    RZLB1 = slope * (RYLB - face2[2, 1]) + face2[2, 2]
            
            if RYUB < face2[2, 1]:
                slope = (face2[2, 2] - face2[0, 2]) / (face2[2, 1] - face2[0, 1])
                
                if np.isinf(slope):
                    RZLB2 = face2[2, 2]
                else:
                    RZLB2 = slope * (RYUB - face2[2, 1]) + face2[2, 2]
            else:
                slope = (face2[3, 2] - face2[2, 2]) / (face2[3, 1] - face2[2, 1])
                
                if np.isinf(slope):
                    RZLB2 = face2[2, 2]
                else:
                    RZLB2 = slope * (RYUB - face2[2, 1]) + face2[2, 2]
            
            if RZUB1 > lc:
                RZUB1 = lc
            if RZUB2 > lc:
                RZUB2 = lc
            if RZLB1 < -lc:
                RZLB1 = -lc
            if RZLB2 < -lc:
                RZLB2 = -lc
            
            if RZUB1 >= -reg_tol and RZUB2 >= -reg_tol and RZLB1 <= reg_tol and RZLB2 <= reg_tol:
                region[regioncount, 0, 0] = RYLB
                region[regioncount, 0, 1] = RZUB1
                region[regioncount, 1, 0] = RYUB
                region[regioncount, 1, 1] = RZUB2
                region[regioncount, 2, 0] = RYLB
                region[regioncount, 2, 1] = RZLB1
                region[regioncount, 3, 0] = RYUB
                region[regioncount, 3, 1] = RZLB2
                regioncount += 1
    else:
        region = np.array([])
    
    return region

def calc_ver(Particel2VerticesEdge, CubeSideLength, catr1, atre1, catr2, atre2, crep, repe, AtomSigma, AtomDensity, Epsilon, lc):
    d2 = []
    distsave = np.zeros((200, 1))
    for i in range(27):
        distsave[i, 0] = 0.7 + (i + 1 - 1) * 0.05
    
    dcutter = 0.4 * lc * 2
    if dcutter > 2.5:
        dres = (dcutter - 2.1) / 173
        for i in range(27, 200):
            distsave[i, 0] = 2.1 + dres * (i + 1 - 28)
        ndist = 200
    else:
        ndist = 27
    # print('Particel2VerticesEdge',Particel2VerticesEdge)
    facever = np.array([[1, 4, 3, 2], [3, 4, 8, 7], [5, 8, 7, 6], [1, 5, 6, 2], [2, 3, 7, 6], [1, 4, 8, 5]])
    facever = facever-1
    fid2d = np.array([[1, 4], [1, 6], [4, 6], [1, 5], [4, 5], [1, 2], [2, 5], [2, 6], [3, 4], [3, 6], [3, 5], [2, 3]])
    fid2d = fid2d-1
    fid3d = np.array([[1, 4, 6], [1, 4, 5], [1, 2, 5], [1, 2, 6], [3, 4, 6], [3, 4, 5], [2, 3, 5], [2, 3, 6]])
    fid3d = fid3d-1    
    # Convert vertices with real x in Angstrom to vertices with x on facet but y, z, and orientation are real
    min_first_item_index = np.argmin(np.array(Particel2VerticesEdge)[:, 0])
    smallest_subarray = Particel2VerticesEdge[min_first_item_index]
    minVertexEdgeX = smallest_subarray[0]
    minVertexEdgeY = smallest_subarray[1]
    minVertexEdgeZ = smallest_subarray[2]
    minVertexIndex = min_first_item_index
    minVertexCount = 0
    minVertexIndices = []
    for i in range(8):
        # if Particel2VerticesEdge[i][0] == minVertexEdgeX: # a more strict criteria 
        if round(Particel2VerticesEdge[i][0],0) == round(minVertexEdgeX,0):
            minVertexCount += 1
            minVertexIndices.append(i)
        
    ds = minVertexEdgeX - CubeSideLength / 2 + AtomSigma
    # flog.write(f'ds = {ds}\n')
    d2.append(ds)
    d2.append(0)
    d2.append(0)
    # VertexWithCorrectYZandOrientation = np.array(Particel2VerticesEdge) - np.array(d2)
    VertexWithCorrectYZandOrientation = np.array(Particel2VerticesEdge)
    # Convert vertices with real y, z, and orientation in Angstrom to vertices with real y, z, and orientation in atomic length
    VertexWithRealYZandOrientationIn1 = VertexWithCorrectYZandOrientation / (CubeSideLength / 2)
    VertexWithRealYZandOrientationInAtomicLength = VertexWithRealYZandOrientationIn1 * lc
    VertexAfterRotationInAtomicLength = VertexWithRealYZandOrientationInAtomicLength
    
    if minVertexCount == 4:
        # print('face-face')
        rcount = 0
        tempv = np.zeros((4, 3))
        for i in range(4):
            tempv[i,:] = VertexAfterRotationInAtomicLength[minVertexIndices[i],:]

        facevs = tempv[np.argsort(tempv[:, 2])]

        if facevs[2, 1] > facevs[3, 1]:
            facev = np.vstack((facevs[3], facevs[2], facevs[0], facevs[1]))
        else:
            facev = np.vstack((facevs[2], facevs[3], facevs[1], facevs[0]))

        if facevs[0, 1] > facevs[1, 1]:
            facev[2] = facevs[0]
            facev[3] = facevs[1]
        else:
            facev[2] = facevs[1]
            facev[3] = facevs[0]

        region = region_calc(facev, CubeSideLength, AtomSigma, lc)
        area = 0
        
        uatrs = np.zeros((region.shape[0], ndist))
        ureps = np.zeros((region.shape[0], ndist))
        utots = np.zeros((region.shape[0], ndist))
        param = np.zeros((ndist, 10))

        ucas = np.zeros(region.shape[0])
        ucrs = np.zeros(region.shape[0])
        ucts = np.zeros(region.shape[0])
        paramc = np.zeros((1, 10))

        for i in range(region.shape[0]):
            rcount = rcount + 1
            tempregion = region[i].copy()
            
            for i3 in range(ndist):
                tempface = facev.copy()
                tempface[:, 0] = tempface[:, 0] + distsave[i3]
                uatrs[i, i3], ureps[i, i3], utots[i, i3], param[i3, :] = calc_gen(tempregion, tempface, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon)
                if abs(param[i3, 6] - distsave[i3, 0]) > 0.1:
                    debugger = 1
            
            tempface = facev.copy()
            tempface[:, 0] = tempface[:, 0] + ds / AtomSigma
            ucas[i], ucrs[i], ucts[i], paramc[:, :] = calc_gen(tempregion, tempface, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon)
        
        # uatr = np.zeros(ndist)
        # urep = np.zeros(ndist)
        # utot = np.zeros(ndist)
        
        # for i3 in range(ndist):
        #     for i in range(region.shape[0]):
        #         uatr[i3] += uatrs[i, i3]
        #         urep[i3] += ureps[i, i3]
        #         utot[i3] += utots[i, i3]
        
        uca = 0
        ucr = 0
        uct = 0
        
        for i in range(region.shape[0]):
            # if not math.isnan(ucas[i]):
            #     uca += ucas[i]
            # if not math.isnan(ucrs[i]):    
            #     ucr += ucrs[i]
            if not math.isnan(ucts[i]):
                uct += ucts[i]
    
    # elif minVertexCount == 2:
    elif minVertexCount == 2  and -CubeSideLength/2 < minVertexEdgeY < CubeSideLength/2 and -CubeSideLength/2 < minVertexEdgeZ < CubeSideLength/2:
        # print('edge-face')
        rcount = 0
        fid = -1
        if minVertexIndices[0] == 0:
            if minVertexIndices[1] == 1:
                fid = 0
            elif minVertexIndices[1] == 3:
                fid = 1
            elif minVertexIndices[1] == 4:
                fid = 2
        elif minVertexIndices[0] == 1:
            if minVertexIndices[1] == 2:
                fid = 3
            elif minVertexIndices[1] == 5:
                fid = 4
        elif minVertexIndices[0] == 2:
            if minVertexIndices[1] == 3:
                fid = 5
            elif minVertexIndices[1] == 6:
                fid = 6
        elif minVertexIndices[0] == 3:
            if minVertexIndices[1] == 7:
                fid = 7
        elif minVertexIndices[0] == 4:
            if minVertexIndices[1] == 5:
                fid = 8
            elif minVertexIndices[1] == 7:
                fid = 9
        elif minVertexIndices[0] == 5:
            if minVertexIndices[1] == 6:
                fid = 10
        elif minVertexIndices[0] == 6:
            if minVertexIndices[1] == 7:
                fid = 11
        
        # uatr = np.zeros(ndist)
        # urep = np.zeros(ndist)
        # utot = np.zeros(ndist)
        uca = 0
        ucr = 0
        uct = 0
        tempv = np.zeros((4, 3))
        facev = np.zeros((4, 3))
        
        for i1 in range(2):
            for i2 in range(4):
                tempv[i2] = VertexAfterRotationInAtomicLength[facever[fid2d[fid, i1], i2]]
            facevs = tempv[np.argsort(tempv[:, 2])]
            if facevs[2][1] > facevs[3][1]:
                facev[0] = facevs[3]
                facev[1] = facevs[2]
            else:
                facev[0] = facevs[2]
                facev[1] = facevs[3]

            if facevs[0][1] > facevs[1][1]:
                facev[2] = facevs[0]
                facev[3] = facevs[1]
            else:
                facev[2] = facevs[1]
                facev[3] = facevs[0]

            
            region = region_calc(facev, CubeSideLength, AtomSigma, lc)
            ucas = np.zeros(region.shape[0])
            ucrs = np.zeros(region.shape[0])
            ucts = np.zeros(region.shape[0])
            paramc = np.zeros((region.shape[0],10))
            uatrs = np.zeros((region.shape[0], ndist))
            ureps = np.zeros((region.shape[0], ndist))
            utots = np.zeros((region.shape[0], ndist))
            param = np.zeros((ndist, 10))
            for i in range(region.shape[0]):
                rcount += 1
                tempregion = region[i]
                
                for i3 in range(ndist):
                    tempface = facev.copy()
                    tempface[:, 0] += distsave[i3][0]
                    uatrs[i][i3], ureps[i][i3], utots[i][i3], param[i3] = calc_gen(tempregion, tempface, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon)
                
                tempface = facev.copy()
                tempface[:, 0] += (ds  + AtomSigma) / AtomSigma # add AtomSigma to correct the gap between the two cubes
                ucas[i], ucrs[i], ucts[i], paramc[i] = calc_gen(tempregion, tempface, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon)
                
            # for i3 in range(ndist):
            #     for i in range(region.shape[0]):
            #         uatr[i3] += uatrs[i, i3]
            #         urep[i3] += ureps[i, i3]
            #         utot[i3] += utots[i, i3]
        
            for i in range(region.shape[0]):
                # if not math.isnan(ucas[i]):
                #     uca += ucas[i]
                # if not math.isnan(ucrs[i]):    
                #     ucr += ucrs[i]
                if not math.isnan(ucts[i]):
                    uct += ucts[i]
            
    # elif minVertexCount == 1:
    elif minVertexCount == 1 and -CubeSideLength/2 < minVertexEdgeY < CubeSideLength/2 and -CubeSideLength/2 < minVertexEdgeZ < CubeSideLength/2:
        # print('vertex-face')
        rcount = 0
        uatr = np.zeros(ndist)
        urep = np.zeros(ndist)
        utot = np.zeros(ndist)
        uca = 0
        ucr = 0
        uct = 0
        tempv = np.zeros((4, 3))
        facev = np.zeros((4, 3))
        # print('fid3d',fid3d)
        for i1 in range(3):
            for i2 in range(4):                
                # original, need to change back # 
                tempv[i2] = VertexAfterRotationInAtomicLength[facever[fid3d[minVertexIndices[0], i1], i2] ]

            facevs = tempv[np.argsort(tempv[:, 2])]
            facevs = tempv[np.argsort(tempv[:, 2])]
            # print('facevs argsort',facevs)
            if facevs[2, 1] > facevs[3][1]:
                facev[0] = facevs[3]
                facev[1] = facevs[2]
            else:
                facev[0] = facevs[2]
                facev[1] = facevs[3]
            if facevs[0, 1] > facevs[1][1]:
                facev[2] = facevs[0]
                facev[3] = facevs[1]
            else:
                facev[2] = facevs[1]
                facev[3] = facevs[0]
            # print('facev',facev)
            region = region_calc(facev, CubeSideLength, AtomSigma, lc)
            ucas = np.zeros(region.shape[0])
            ucrs = np.zeros(region.shape[0])
            ucts = np.zeros(region.shape[0])
            paramc = np.zeros((region.shape[0],10))
            uatrs = np.zeros((region.shape[0], ndist))
            ureps = np.zeros((region.shape[0], ndist))
            utots = np.zeros((region.shape[0], ndist))
            param = np.zeros((ndist, 10))
            for i in range(region.shape[0]):
                rcount += 1
                tempregion = region[i]
                for i3 in range(ndist):
                    tempface = facev.copy()
                    tempface[:, 0] += distsave[i3][0]
                    uatrs[i][i3], ureps[i][i3], utots[i][i3], param[i3] = calc_gen(tempregion, tempface, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon)
                    # if abs(param[i3][6] - distsave[i3][0]) > 0.1:
                    #     debugger = 1
                
                tempface = facev.copy()
                tempface[:, 0] += (ds + AtomSigma) / AtomSigma # add AtomSigma to correct the gap between the two cubes
                ucas[i], ucrs[i], ucts[i], paramc[i] = calc_gen(tempregion, tempface, AtomSigma, lc, catr1, atre1, catr2, atre2, crep, repe, Epsilon)
        
            # for i3 in range(ndist):
            #     for i in range(region.shape[0]):
            #         uatr[i3] += uatrs[i][i3]
            #         urep[i3] += ureps[i][i3]
            #         utot[i3] += utots[i][i3]
        
            for i in range(region.shape[0]):
                if not math.isnan(ucts[i]):
                    uct += ucts[i]
                # if not math.isnan(ucas[i]):
                #     uca += ucas[i]
                # if not math.isnan(ucrs[i]):    
                #     ucr += ucrs[i]
                
    else: # case 4, edge-edge, which was not developed by Brian Lee in his study
          # without this "else", the edge-edge is classified as vertex-face case, which is incorrect 
        # print('edge-edge')
        uct = 0
        ds = 0
        
    return uct, ds
######################## vdW energy calculation functions ##########################
################################################################################################

################################################################################################
######################## ML energy calculation functions ##########################
def GetOrientationAndDistance(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ):
    Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ = Reorientation(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)
    AverageModelSize = (Size1 + Size2)/2
    P1P2 = np.array(Particle2Centroid) - np.array(Particle1Centroid)
    
    X = abs(P1P2[0])/AverageModelSize # Normalized x distance between P1 and P2 
    Y = abs(P1P2[1])/AverageModelSize # Normalized y distance between P1 and P2 
    Z = abs(P1P2[2])/AverageModelSize # Normalized z distance between P1 and P2 
    
    # Normalize the vectors
    x1 = Particle1VectorsX / np.linalg.norm(Particle1VectorsX)
    y1 = Particle1VectorsY / np.linalg.norm(Particle1VectorsY)
    z1 = Particle1VectorsZ / np.linalg.norm(Particle1VectorsZ)
    x2 = Particle2VectorsX / np.linalg.norm(Particle2VectorsX)
    y2 = Particle2VectorsY / np.linalg.norm(Particle2VectorsY)
    z2 = Particle2VectorsZ / np.linalg.norm(Particle2VectorsZ)
    # Generate rotation matrix
    R_mat1 = np.vstack((x1, y1, z1))
    R_mat2 = np.vstack((x2, y2, z2))                                                               
    # Convert the rotation matrix to a scipy Rotation object
    r1 = R.from_matrix(R_mat1)
    r2 = R.from_matrix(R_mat2)
    # Get the Euler angles
    euler_angles1 = r1.as_euler('zxz', degrees=True)
    euler_angles2 = r2.as_euler('zxz', degrees=True)
    # print(f'alpha: {euler_angles1[0]}, beta: {euler_angles1[1]}, gamma: {euler_angles1[2]}')
    # print(f'alpha: {euler_angles2[0]}, beta: {euler_angles2[1]}, gamma: {euler_angles2[2]}')
    Alpha = euler_angles2[0] - euler_angles1[0]
    Beta = euler_angles2[1] - euler_angles1[1]
    Gamma = euler_angles2[2] - euler_angles1[2]
    # wrap angles back to 0-90 degrees according to the training data
    Alpha = wrapAngle(Alpha)
    Beta = wrapAngle(Beta)
    Gamma = wrapAngle(Gamma)
    # print('Size1:',Size1)
    # print('Size2:',Size2)
    # print('Alpha:', wrapAngle(Alpha))
    # print('Beta:', wrapAngle(Beta))
    # print('Gamma:', wrapAngle(Gamma))
    # print('X:', X)
    # print('Y:', Y)
    # print('Z:', Z)
    # print('\n \n\n')
    return Size1, Size2, Alpha, Beta, Gamma, X, Y, Z
######################## ML energy calculation functions ##########################
################################################################################################

################################################################################################
######################## Shared common functions ##########################
def CubeRotation(CubeCentroid, CubeVector, RotationCenter, RotationAngles):
    # print('RotationAngles',RotationAngles)
    # print('CubeCentroid',CubeCentroid)
    # print('RotationCenter',RotationCenter)
    VectorX = np.array(CubeVector[0])
    VectorY = np.array(CubeVector[1])
    VectorZ = np.array(CubeVector[2])
    CubeCentroid = np.array(CubeCentroid)
    RotationCenter = np.array(RotationCenter)
    Alpha = RotationAngles[0]/180*np.pi
    Beta = RotationAngles[1]/180*np.pi
    Gamma = RotationAngles[2]/180*np.pi
    RotationX = np.array([[1, 0, 0], [0, math.cos(Alpha), -math.sin(Alpha)], [0, math.sin(Alpha), math.cos(Alpha)]])
    RotationY = np.array([[math.cos(Beta), 0, math.sin(Beta)], [0, 1, 0], [-math.sin(Beta), 0, math.cos(Beta)]])
    RotationZ = np.array([[math.cos(Gamma), -math.sin(Gamma), 0], [math.sin(Gamma), math.cos(Gamma), 0], [0, 0, 1]])  
    # find out the vectors from the rotation center to the cube vectors' end points
    RotationCenter_To_VectorXEndPoint = VectorX + CubeCentroid - RotationCenter
    RotationCenter_To_VectorYEndPoint = VectorY + CubeCentroid - RotationCenter
    RotationCenter_To_VectorZEndPoint = VectorZ + CubeCentroid - RotationCenter
    # rotation center to cube centroid
    RotationCenter_To_CubeCentroid = CubeCentroid - RotationCenter
    # rotate end points of vectors about the rotation center
    RotationCenter_To_VectorXEndPoint = RotationCenter_To_VectorXEndPoint @ RotationX @ RotationY @ RotationZ
    RotationCenter_To_VectorYEndPoint = RotationCenter_To_VectorYEndPoint @ RotationX @ RotationY @ RotationZ
    RotationCenter_To_VectorZEndPoint = RotationCenter_To_VectorZEndPoint @ RotationX @ RotationY @ RotationZ
    # rotate cube centroid about the rotation center
    RotationCenter_To_CubeCentroid = RotationCenter_To_CubeCentroid @ RotationX @ RotationY @ RotationZ
    # print('RotationCenter_To_CubeCentroid')
    # print(RotationCenter_To_CubeCentroid)
    # convert back to the cube vectors
    VectorX = RotationCenter_To_VectorXEndPoint - RotationCenter_To_CubeCentroid
    VectorY = RotationCenter_To_VectorYEndPoint - RotationCenter_To_CubeCentroid
    VectorZ = RotationCenter_To_VectorZEndPoint - RotationCenter_To_CubeCentroid
    # normalize cube vectors to unit length
    VectorX = VectorX / np.linalg.norm(VectorX)
    VectorY = VectorY / np.linalg.norm(VectorY)
    VectorZ = VectorZ / np.linalg.norm(VectorZ)
    
    CubeCentroid = RotationCenter_To_CubeCentroid + CubeCentroid
    Vector = np.array([VectorX,VectorY,VectorZ])
    # print('CubeCentroid', 'Vector')
    # print(CubeCentroid, Vector)
    return CubeCentroid, Vector

def InitialConfiguration(NumberOfParticles, BoxLength, AtomSigma, SizeRatioDictionary):
    Sizes = np.zeros(NumberOfParticles) # N (NxNxN)
    
    keys_list = list(SizeRatioDictionary.keys())
    values_list = list(SizeRatioDictionary.values())
    valuesSum = sum(values_list)
    ParticleID = 0
    CumulativeNumberOfParticlesList = []
    for size in keys_list:
        NumberOfCurrentSize = round(SizeRatioDictionary[size] / valuesSum * NumberOfParticles)
        CumulativeNumberOfParticlesList.append(NumberOfCurrentSize)
    ToAdd = NumberOfParticles - sum(CumulativeNumberOfParticlesList)
    for size in keys_list: 
        if ParticleID <= NumberOfParticles:
            for i in range(NumberOfCurrentSize):
                Sizes[ParticleID] = size
                ParticleID += 1
    for ToAddIndex in range(ToAdd): # sometimes the last particle is not assigned a size due to the round function above, here this issue is fixed by assigning the last size in the size list to that particle
         ParticleID = ParticleID + ToAddIndex
         Sizes[ParticleID] = 12
    
    # initialization of arrays
    Rx = np.zeros(NumberOfParticles)  # Angstrom
    Ry = np.zeros(NumberOfParticles)  # Angstrom
    Rz = np.zeros(NumberOfParticles)  # Angstrom
    
    VectorX = np.array([1.0, 0.0, 0.0] * NumberOfParticles).reshape(NumberOfParticles, 3)  # orientation of particles
    VectorY = np.array([0.0, 1.0, 0.0] * NumberOfParticles).reshape(NumberOfParticles, 3)  # orientation of particles
    VectorZ = np.array([0.0, 0.0, 1.0] * NumberOfParticles).reshape(NumberOfParticles, 3)  # orientation of particles
    
    BoxLengthHalf = BoxLength / 2.0  # Angstrom
    SigmaCut = 2 * AtomSigma  # sigma is the size of the atoms
    SigmaCutSquare = SigmaCut**2  # the square of SigmaCut, % Angstrom

    FirstPositionX = np.random.uniform(0, 1)
    FirstPositionY = np.random.uniform(0, 1)
    FirstPositionZ = np.random.uniform(0, 1)
    FirstCubeAlpha = np.random.uniform(0, 90)
    FirstCubeBeta = np.random.uniform(0, 90)
    FirstCubeGamma = np.random.uniform(0, 90)
    
    FirstCubePosition, FirstCubeVectors = CubeRotation([FirstPositionX, FirstPositionY, FirstPositionZ], [VectorX[0],VectorY[0],VectorZ[0]], [FirstPositionX, FirstPositionY, FirstPositionZ], [FirstCubeAlpha,FirstCubeBeta,FirstCubeGamma])
    VectorX[0] = FirstCubeVectors[0]
    VectorY[0] = FirstCubeVectors[1]
    VectorZ[0] = FirstCubeVectors[2]

    Rx[0] = BoxLength * FirstPositionX - BoxLengthHalf  # Angstrom
    Ry[0] = BoxLength * FirstPositionY - BoxLengthHalf  # Angstrom
    Rz[0] = BoxLength * FirstPositionZ - BoxLengthHalf  # Angstrom
    
    
    for i in range(1, NumberOfParticles):
        print(f'Insertion Of Molecule {i+1} Successful')
        Repeat = True
        while Repeat:
            Repeat = False
            # random position
            PositionX = BoxLength * np.random.uniform(0, 1) - BoxLengthHalf
            PositionY = BoxLength * np.random.uniform(0, 1) - BoxLengthHalf
            PositionZ = BoxLength * np.random.uniform(0, 1) - BoxLengthHalf
            # random rotation
            Alpha = np.random.uniform(0, 90)
            Beta = np.random.uniform(0, 90)
            Gamma = np.random.uniform(0, 90)
            _, CubeVectors = CubeRotation([PositionX, PositionY, PositionZ], [VectorX[i],VectorY[i],VectorZ[i]], [PositionX, PositionY, PositionZ], [Alpha,Beta,Gamma])
            CubeSideLength1 = Sizes[i] * AASigma
            for K in range(i):
                CubeSideLength2 = Sizes[K] * AASigma
                DistanceFromAnOldAtomX = PositionX - Rx[K]
                DistanceFromAnOldAtomY = PositionY - Ry[K]
                DistanceFromAnOldAtomZ = PositionZ - Rz[K]
                DistanceFromAnOldAtomX -= BoxLength * round(DistanceFromAnOldAtomX / BoxLength)
                DistanceFromAnOldAtomY -= BoxLength * round(DistanceFromAnOldAtomY / BoxLength)
                DistanceFromAnOldAtomZ -= BoxLength * round(DistanceFromAnOldAtomZ / BoxLength)
                DistanceSquare = DistanceFromAnOldAtomX**2 + DistanceFromAnOldAtomY**2 + DistanceFromAnOldAtomZ**2
                # 0.5*sqrt(3)*CubeSideLength is considered as the radius of a cube, used for determining overlap
                if DistanceSquare < (0.866 * (CubeSideLength1 + CubeSideLength2))**2:
                    Repeat = True
                    break
        VectorX[i] = CubeVectors[0]
        VectorY[i] = CubeVectors[1]
        VectorZ[i] = CubeVectors[2]
        Rx[i] = PositionX
        Ry[i] = PositionY
        Rz[i] = PositionZ
    
    restartStep = 0
    print("Start from randam configuration")
    
    LAMMPSTrajectoryFile = open('LAMMPSTrajectory.lammpstrj', 'w')  # positions in movie format every TrajectoryInterval steps
    EnergyFile = open('Energies.out', 'w')  # stores energies every EnergyOutputInterval steps
    TimeFile = open('Time.out', 'w')  # stores energies every EnergyOutputInterval steps

    return Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ, restartStep, LAMMPSTrajectoryFile, EnergyFile, TimeFile

def EnergyBetweenTwoParticles(i, j, Size_i, Size_j, ParticleiCentroid, ParticlejCentroid, VectorX_i, VectorY_i, VectorZ_i, VectorX_j, VectorY_j, VectorZ_j):
    if model == 'AACG':
        ############################################################################################
        ####################### AA/CG LJ energy calculation ###################
        # cube 1
        COMX,COMY,COMZ = ParticleiCentroid[0],ParticleiCentroid[1],ParticleiCentroid[2]
        VectorX, VectorY, VectorZ = VectorX_i, VectorY_i, VectorZ_i
        atomList1 = findAtomPositionsInCube(CGCubeSize,CGSigma,COMX,COMY,COMZ,VectorX, VectorY, VectorZ)
        # cube 2
        COMX,COMY,COMZ = ParticlejCentroid[0],ParticlejCentroid[1],ParticlejCentroid[2]
        VectorX, VectorY, VectorZ = VectorX_j, VectorY_j, VectorZ_j
        atomList2 = findAtomPositionsInCube(CGCubeSize,CGSigma,COMX,COMY,COMZ,VectorX, VectorY, VectorZ)
        # Energy
        CubeCubeEnergy = LJPotentialBetweenTwoCubes(CGSigma, CGEpsilon, cutoffCG, atomList1, atomList2)
        ####################### AA/CG LJ energy calculation ###################
        ############################################################################################
    
    elif model == 'vdW':
        ############################################################################################
        ####################### vdW energy calculation ###################
        Size1, Size1, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ = Reorientation(i, j, Size_i, Size_j, ParticleiCentroid, ParticlejCentroid, VectorX_i, VectorY_i, VectorZ_i, VectorX_j, VectorY_j, VectorZ_j)
        CubeSideLength = Size1 * AASigma # Size1 and Size2 have to be the same in the vdW model 
        Particel2VerticesEdge = findVerticesOfCube(CubeSideLength,Particle2Centroid[0],Particle2Centroid[1],Particle2Centroid[2],Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)
        # Parameters
        catr1 = -7.75596  # c of short range attraction
        atre1 = 3.4339  # n of short range attraction
        catr2 = -4.65093  # c of long range attraction
        atre2 = 2.7167  # n of long range attraction
        crep = 3.85898  # C of repulsion
        repe = 10.66482  # n of repulsion
        lc = math.floor(CubeSideLength / AASigma) / 2
        Epsilon = (Hamaker * (10 ** -19) / (4 * math.pi ** 2 * AtomDensity ** 2 * AASigma ** 6 * 6.9477 * (10 ** -21)))  # Energy parameter Epsilon
        # Call function to calculate potential energy between two particles
        CubeCubeEnergy, ds = calc_ver(Particel2VerticesEdge, CubeSideLength, catr1, atre1, catr2,atre2,crep,repe, AASigma, AtomDensity, Epsilon, lc)
        if math.isinf(CubeCubeEnergy):
            CubeCubeEnergy = 0
        ####################### vdW energy calculation ###################
        ############################################################################################

    elif model == 'ML':
        ############################################################################################
        ####################### ML energy calculation ###################
        Size1, Size2, Alpha, Beta, Gamma, X, Y, Z = GetOrientationAndDistance(i, j, Size_i, Size_j, ParticleiCentroid, ParticlejCentroid, VectorX_i, VectorY_i, VectorZ_i, VectorX_j, VectorY_j, VectorZ_j)
        AverageModelSize = (Size1 + Size2)/2
        # convert a row of descriptors to dataframe for prediction
        my_FeatureList = [[Size1, Size2, Alpha, Beta, Gamma, X, Y, Z]]
        df_ForEnergy = pd.DataFrame(my_FeatureList, columns=['Model1Size','Model2Size','Alpha','Beta','Gamma','X','Y','Z'])
        df_ForType = df_ForEnergy.drop(['X'], axis='columns') 
        # predict type for the energy curve 
        TypePredicted = MLModel_ForType.predict(df_ForType)[0]
        # print(TypePredicted)
        # predict energy value
        if TypePredicted == 0:
            EnergyPredicted = MLModel_ForEnergy_Type0.predict(df_ForEnergy)[0]
            # print('Type0', EnergyPredicted)     
        else:
            EnergyPredicted = MLModel_ForEnergy_Type1.predict(df_ForEnergy)[0]
            # print('Type1', EnergyPredicted)     
        
        EnergyPredicted = EnergyPredicted * AverageModelSize # energies in the models are normalized, so now they need to be convert back to real values
        # print('Size1:',Size1)
        # print('Size2:',Size2)
        # print('Alpha:', wrapAngle(Alpha))
        # print('Beta:', wrapAngle(Beta))
        # print('Gamma:', wrapAngle(Gamma))
        # print('X:', X)
        # print('Y:', Y)
        # print('Z:', Z)
        # print('Energy: ',EnergyPredicted)
        # print('\n \n\n')
        CubeCubeEnergy = EnergyPredicted * EnergyScaleFactor
        ####################### ML energy calculation ###################
        ############################################################################################
    
    return CubeCubeEnergy

def OneParticleEnergy(i, Size_i, Rx_i, Ry_i, Rz_i, Sizes, Rx, Ry, Rz, VectorX_i, VectorY_i, VectorZ_i, VectorX, VectorY, VectorZ, NumberOfParticles,clusterParticleList):    
    overLapFlag = False
    CurrentAtomTotalPotentialEnergy = 0
    for j in range(NumberOfParticles):
        if j != i and j not in clusterParticleList:             
            Size_i = Sizes[i]
            Size_j = Sizes[j]
            VectorX_j = VectorX[j, :]
            VectorY_j = VectorY[j, :]
            VectorZ_j = VectorZ[j, :]
            Rx_j = Rx[j]  # Angstrom
            Rx_ij = Rx_i - Rx_j  # Angstrom
            Ry_j = Ry[j]  # Angstrom
            Ry_ij = Ry_i - Ry_j  # Angstrom
            Rz_j = Rz[j]  # Angstrom
            Rz_ij = Rz_i - Rz_j  # Angstrom
            Rx_ij = Rx_ij - BoxLength * round(Rx_ij / BoxLength)  # Angstrom
            Ry_ij = Ry_ij - BoxLength * round(Ry_ij / BoxLength)  # Angstrom
            Rz_ij = Rz_ij - BoxLength * round(Rz_ij / BoxLength)  # Angstrom
            RijSquare = Rx_ij**2 + Ry_ij**2 + Rz_ij**2  # Angstrom^2
            ParticleiCentroid = np.array([0, 0, 0])
            ParticlejCentroid = np.array([-Rx_ij, -Ry_ij, -Rz_ij])
            
            MaxSize = max(Size_i, Size_j)
            CutOff =  2 * MaxSize * AASigma  # Angstrom 
            CutOffSquare = CutOff**2  # Angstrom^2
            
            if RijSquare < CutOffSquare:
                Size1 = Size_i
                Size2 = Size_j
                CubeSideLength1 = Size1 * AASigma
                CubeSideLength2 = Size2 * AASigma
                Particle1Centroid = ParticleiCentroid
                Particle2Centroid = ParticlejCentroid
                Particle1VectorsX = VectorX_i
                Particle1VectorsY = VectorY_i
                Particle1VectorsZ = VectorZ_i
                Particle2VectorsX = VectorX_j
                Particle2VectorsY = VectorY_j
                Particle2VectorsZ = VectorZ_j

                if CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Particle1Centroid[0],Particle1Centroid[1],Particle1Centroid[2],Particle1VectorsX,Particle1VectorsY,Particle1VectorsZ,Particle2Centroid[0],Particle2Centroid[1],Particle2Centroid[2],Particle2VectorsX,Particle2VectorsY,Particle2VectorsZ):
                    # print("Overlap. CurrentPair.")
                    CurrentPairPotentialEnergy = 1000 # just a large number
                    overLapFlag = True
                    # print(f"{i} and {j} overlap")
                    flog.write(f"{i} and {j} overlap\n")
                    flog.write(f'{CubeSideLength1}, {CubeSideLength2},{Particle1Centroid},{Particle1VectorsX},{Particle1VectorsY},{Particle1VectorsZ},{Particle2Centroid},{Particle2VectorsX},{Particle2VectorsY},{Particle2VectorsZ}\n')
                    # print(CubeSideLength1, CubeSideLength2,Particle1Centroid[0],Particle1Centroid[1],Particle1Centroid[2],Particle1VectorsX,Particle1VectorsY,Particle1VectorsZ,Particle2Centroid[0],Particle2Centroid[1],Particle2Centroid[2],Particle2VectorsX,Particle2VectorsY,Particle2VectorsZ)
                else:
                    # start_time = time.time()
                    # Size1, Size2, Alpha, Beta, Gamma, X, Y, Z = GetOrientationAndDistance(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)                
                    CurrentPairPotentialEnergy = EnergyBetweenTwoParticles(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)  # kcal/mol
                    # end_time = time.time()
                    # elapsed_time = end_time - start_time
                    # print(f"Elapsed time: {elapsed_time} seconds")
                CurrentAtomTotalPotentialEnergy += CurrentPairPotentialEnergy  # kcal/mol
                
    return CurrentAtomTotalPotentialEnergy,overLapFlag

# def OneParticleEnergy(i, Size_i, Rx_i, Ry_i, Rz_i, Sizes, Rx, Ry, Rz, VectorX_i, VectorY_i, VectorZ_i, VectorX, VectorY, VectorZ, NumberOfParticles,clusterParticleList):    
#     overLapFlag = False
#     CurrentAtomTotalPotentialEnergy = 0
#     j_tempList = []
#     for j in range(NumberOfParticles):
#         if j != i and j not in clusterParticleList:
#                     j_tempList.append(j)
#     pairs = [(i, j) for j in j_tempList]
#     # print('pairs', pairs)
#     partial_do_parallel_OneParticleEnergy_calculation = partial(do_parallel_OneParticleEnergy_calculation, 
#                                                                 Size_i=Size_i, Rx_i=Rx_i, Ry_i=Ry_i, Rz_i=Rz_i, Sizes=Sizes, Rx=Rx, Ry=Ry, Rz=Rz, VectorX_i=VectorX_i, VectorY_i=VectorY_i, VectorZ_i=VectorZ_i, 
#                                                                 VectorX=VectorX, VectorY=VectorY, VectorZ=VectorZ, NumberOfParticles=NumberOfParticles, clusterParticleList=clusterParticleList)
#     with Pool(Cores) as pool:
#         pair_energies = pool.starmap(partial_do_parallel_OneParticleEnergy_calculation, pairs)
    
#     CurrentAtomTotalPotentialEnergy = sum(pair_energies)
    
#     return CurrentAtomTotalPotentialEnergy, False

# def do_parallel_OneParticleEnergy_calculation(i, j, Size_i, Rx_i, Ry_i, Rz_i, Sizes, Rx, Ry, Rz, VectorX_i, VectorY_i, VectorZ_i, VectorX, VectorY, VectorZ, NumberOfParticles,clusterParticleList):    
#     print('pair', i, j)
#     overLapFlag = False
#     CurrentAtomTotalPotentialEnergy = 0
            
#     Size_i = Sizes[i]
#     Size_j = Sizes[j]
#     VectorX_j = VectorX[j, :]
#     VectorY_j = VectorY[j, :]
#     VectorZ_j = VectorZ[j, :]
#     Rx_j = Rx[j]  # Angstrom
#     Rx_ij = Rx_i - Rx_j  # Angstrom
#     Ry_j = Ry[j]  # Angstrom
#     Ry_ij = Ry_i - Ry_j  # Angstrom
#     Rz_j = Rz[j]  # Angstrom
#     Rz_ij = Rz_i - Rz_j  # Angstrom
#     Rx_ij = Rx_ij - BoxLength * round(Rx_ij / BoxLength)  # Angstrom
#     Ry_ij = Ry_ij - BoxLength * round(Ry_ij / BoxLength)  # Angstrom
#     Rz_ij = Rz_ij - BoxLength * round(Rz_ij / BoxLength)  # Angstrom
#     RijSquare = Rx_ij**2 + Ry_ij**2 + Rz_ij**2  # Angstrom^2
#     ParticleiCentroid = np.array([0, 0, 0])
#     ParticlejCentroid = np.array([-Rx_ij, -Ry_ij, -Rz_ij])
    
#     MaxSize = max(Size_i, Size_j)
#     CutOff =  2 * MaxSize * AASigma  # Angstrom 
#     CutOffSquare = CutOff**2  # Angstrom^2
    
#     if RijSquare < CutOffSquare:
#         Size1 = Size_i
#         Size2 = Size_j
#         CubeSideLength1 = Size1 * AASigma
#         CubeSideLength2 = Size2 * AASigma
#         Particle1Centroid = ParticleiCentroid
#         Particle2Centroid = ParticlejCentroid
#         Particle1VectorsX = VectorX_i
#         Particle1VectorsY = VectorY_i
#         Particle1VectorsZ = VectorZ_i
#         Particle2VectorsX = VectorX_j
#         Particle2VectorsY = VectorY_j
#         Particle2VectorsZ = VectorZ_j

#         if CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Particle1Centroid[0],Particle1Centroid[1],Particle1Centroid[2],Particle1VectorsX,Particle1VectorsY,Particle1VectorsZ,Particle2Centroid[0],Particle2Centroid[1],Particle2Centroid[2],Particle2VectorsX,Particle2VectorsY,Particle2VectorsZ):
#             CurrentPairPotentialEnergy = 1000 # just a large number
#             overLapFlag = True
#         else:
#             CurrentPairPotentialEnergy = EnergyBetweenTwoParticles(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)  # kcal/mol
#         CurrentAtomTotalPotentialEnergy += CurrentPairPotentialEnergy  # kcal/mol
                
#     # return CurrentAtomTotalPotentialEnergy, overLapFlag
#     return CurrentAtomTotalPotentialEnergy

# def TotalEnergy(Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ):
#     SystemPotentialEnergy = 0
#     tempList = []
#     for i in range(NumberOfParticles-1):
#         Size_i = Sizes[i]
#         Rx_i = Rx[i]  # Angstrom
#         Ry_i = Ry[i]  # Angstrom
#         Rz_i = Rz[i]  # Angstrom
#         VectorX_i = VectorX[i, :]
#         VectorY_i = VectorY[i, :]
#         VectorZ_i = VectorZ[i, :]
#         for j in range(i+1, NumberOfParticles):
#             tempList.append([i,j])
#             Size_j = Sizes[j]
#             Rx_j = Rx[j]  # Angstrom
#             Ry_j = Ry[j]  # Angstrom
#             Rz_j = Rz[j]  # Angstrom
#             Rx_ij = Rx_i - Rx_j
#             Ry_ij = Ry_i - Ry_j
#             Rz_ij = Rz_i - Rz_j
#             VectorX_j = VectorX[j, :]
#             VectorY_j = VectorY[j, :]
#             VectorZ_j = VectorZ[j, :]        
#             Rx_ij -= BoxLength * round(Rx_ij / BoxLength)  # Angstrom
#             Ry_ij -= BoxLength * round(Ry_ij / BoxLength)  # Angstrom
#             Rz_ij -= BoxLength * round(Rz_ij / BoxLength)  # Angstrom          
#             RijSquare = Rx_ij**2 + Ry_ij**2 + Rz_ij**2  # Angstrom^2  
#             ParticleiCentroid = np.array([0, 0, 0])
#             ParticlejCentroid = np.array([-Rx_ij, -Ry_ij, -Rz_ij])
            
#             MaxSize = max(Size_i, Size_j)
#             CutOff =  2 * MaxSize * AASigma  # Angstrom 
#             CutOffSquare = CutOff**2  # Angstrom^2
            
#             if RijSquare < CutOffSquare:
#                 # if Size_i >= Size_j: # always make larger size one as particle 1 
#                 Size1 = Size_i
#                 Size2 = Size_j
#                 CubeSideLength1 = Size1 * AASigma
#                 CubeSideLength2 = Size2 * AASigma
#                 Particle1Centroid = ParticleiCentroid
#                 Particle2Centroid = ParticlejCentroid
#                 Particle1VectorsX = VectorX_i
#                 Particle1VectorsY = VectorY_i
#                 Particle1VectorsZ = VectorZ_i
#                 Particle2VectorsX = VectorX_j
#                 Particle2VectorsY = VectorY_j
#                 Particle2VectorsZ = VectorZ_j
#                 if CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Rx_i,Ry_i,Rz_i,VectorX_i,VectorY_i,VectorZ_i,Rx_j,Ry_j,Rz_j,VectorX_j,VectorY_j,VectorZ_j):
#                     CurrentPairPotentialEnergy = 100 # just a large number
#                     overLapFlag = True
                    
#                 else:
#                     # Size1, Size2, Alpha, Beta, Gamma, X, Y, Z = GetOrientationAndDistance(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)                    
#                     CurrentPairPotentialEnergy = EnergyBetweenTwoParticles(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)  # kcal/mol
                    
#                 SystemPotentialEnergy = SystemPotentialEnergy + CurrentPairPotentialEnergy  # kcal/mol
#                 print(i,j,CurrentPairPotentialEnergy) 
#     print(tempList) 
#     print(len(tempList)) 
#     return SystemPotentialEnergy


def TotalEnergy(Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ):
    SystemPotentialEnergy = 0
    pairs = [(i, j) for i in range(NumberOfParticles - 1) for j in range(i + 1, NumberOfParticles)]
    partial_do_parallel_TotalEnergy_calculation = partial(do_parallel_TotalEnergy_calculation, Sizes=Sizes, Rx=Rx, Ry=Ry, Rz=Rz,
                                                      VectorX=VectorX, VectorY=VectorY, VectorZ=VectorZ)
    with Pool(Cores) as pool:
        pair_energies = pool.starmap(partial_do_parallel_TotalEnergy_calculation, pairs)
    
    print('pair_energies', pair_energies)
    print('pair_energies len', len(pair_energies))
    aaa = do_parallel_TotalEnergy_calculation(0,1, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ)
    print('aaa',aaa)
    SystemPotentialEnergy = sum(pair_energies)
    
    return SystemPotentialEnergy


def do_parallel_TotalEnergy_calculation(i,j, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ):
    CurrentPairPotentialEnergy = 0
    Size_i = Sizes[i]
    Rx_i = Rx[i]  # Angstrom
    Ry_i = Ry[i]  # Angstrom
    Rz_i = Rz[i]  # Angstrom
    VectorX_i = VectorX[i, :]
    VectorY_i = VectorY[i, :]
    VectorZ_i = VectorZ[i, :]
    Size_j = Sizes[j]
    Rx_j = Rx[j]  # Angstrom
    Ry_j = Ry[j]  # Angstrom
    Rz_j = Rz[j]  # Angstrom
    Rx_ij = Rx_i - Rx_j
    Ry_ij = Ry_i - Ry_j
    Rz_ij = Rz_i - Rz_j
    VectorX_j = VectorX[j, :]
    VectorY_j = VectorY[j, :]
    VectorZ_j = VectorZ[j, :]        
    Rx_ij -= BoxLength * round(Rx_ij / BoxLength)  # Angstrom
    Ry_ij -= BoxLength * round(Ry_ij / BoxLength)  # Angstrom
    Rz_ij -= BoxLength * round(Rz_ij / BoxLength)  # Angstrom          
    RijSquare = Rx_ij**2 + Ry_ij**2 + Rz_ij**2  # Angstrom^2  
    ParticleiCentroid = np.array([0, 0, 0])
    ParticlejCentroid = np.array([-Rx_ij, -Ry_ij, -Rz_ij])
    
    MaxSize = max(Size_i, Size_j)
    CutOff =  2 * MaxSize * AASigma  # Angstrom 
    CutOffSquare = CutOff**2  # Angstrom^2
    
    if RijSquare < CutOffSquare:
        # if Size_i >= Size_j: # always make larger size one as particle 1 
        Size1 = Size_i
        Size2 = Size_j
        CubeSideLength1 = Size1 * AASigma
        CubeSideLength2 = Size2 * AASigma
        Particle1Centroid = ParticleiCentroid
        Particle2Centroid = ParticlejCentroid
        Particle1VectorsX = VectorX_i
        Particle1VectorsY = VectorY_i
        Particle1VectorsZ = VectorZ_i
        Particle2VectorsX = VectorX_j
        Particle2VectorsY = VectorY_j
        Particle2VectorsZ = VectorZ_j
        if CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Rx_i,Ry_i,Rz_i,VectorX_i,VectorY_i,VectorZ_i,Rx_j,Ry_j,Rz_j,VectorX_j,VectorY_j,VectorZ_j):
            CurrentPairPotentialEnergy = 100 # just a large number
            overLapFlag = True
        else:
            # Size1, Size2, Alpha, Beta, Gamma, X, Y, Z = GetOrientationAndDistance(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)                    
            CurrentPairPotentialEnergy = EnergyBetweenTwoParticles(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ)  # kcal/mol
        print(i,j,CurrentPairPotentialEnergy)
        
    return CurrentPairPotentialEnergy

def Reorientation(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ):   
    # Suppress the warning
    warnings.filterwarnings("ignore")
    # if cube 2 size larger than cube 1, then cube 2 as reference
    # or, if cube 1 and cube 2 having the same size, the one with smaller index is considered as reference 
    if (Size1 == Size2 and i > j) or (Size1 < Size2):
        # print('cube 2 as reference')
        Size1Temp = Size1
        Size2Temp = Size2
        Particle1CentroidTemp = Particle1Centroid.copy()
        Particle2CentroidTemp = Particle2Centroid.copy()
        Particle1VectorsXTemp = Particle1VectorsX.copy()
        Particle1VectorsYTemp = Particle1VectorsY.copy()
        Particle1VectorsZTemp = Particle1VectorsZ.copy()
        Particle2VectorsXTemp = Particle2VectorsX.copy()
        Particle2VectorsYTemp = Particle2VectorsY.copy()
        Particle2VectorsZTemp = Particle2VectorsZ.copy()
        Size1 = Size2Temp
        Size2 = Size1Temp
        Particle1Centroid = Particle2CentroidTemp
        Particle2Centroid = Particle1CentroidTemp
        Particle1VectorsX = Particle2VectorsXTemp
        Particle1VectorsY = Particle2VectorsYTemp
        Particle1VectorsZ = Particle2VectorsZTemp
        Particle2VectorsX = Particle1VectorsXTemp
        Particle2VectorsY = Particle1VectorsYTemp
        Particle2VectorsZ = Particle1VectorsZTemp

    # Normalize the vectors of cube 1
    Particle1VectorsX = np.array(Particle1VectorsX) / np.linalg.norm(Particle1VectorsX)
    Particle1VectorsY = np.array(Particle1VectorsY) / np.linalg.norm(Particle1VectorsY)
    Particle1VectorsZ = np.array(Particle1VectorsZ) / np.linalg.norm(Particle1VectorsZ)
    # Generate rotation matrix for cube 1
    R1 = np.vstack((Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ))
    # Get the rotation to standard basis vectors ([1,0,0], [0,1,0], and [0,0,1])
    # Calculate the inverse matrix of cube 1
    inv_1 = np.linalg.inv(R1)
    # Now let's rotate cube B using this inverse rotation
    # particle centroid in new coordinate system, with cube 1 particle as the centroid
    Particle2VectorXEndPoint = [Particle2VectorsX[0] + Particle2Centroid[0] - Particle1Centroid[0], Particle2VectorsX[1] + Particle2Centroid[1] - Particle1Centroid[1], Particle2VectorsX[2] + Particle2Centroid[2] - Particle1Centroid[2]]
    Particle2VectorYEndPoint = [Particle2VectorsY[0] + Particle2Centroid[0] - Particle1Centroid[0], Particle2VectorsY[1] + Particle2Centroid[1] - Particle1Centroid[1], Particle2VectorsY[2] + Particle2Centroid[2] - Particle1Centroid[2]]
    Particle2VectorZEndPoint = [Particle2VectorsZ[0] + Particle2Centroid[0] - Particle1Centroid[0], Particle2VectorsZ[1] + Particle2Centroid[1] - Particle1Centroid[1], Particle2VectorsZ[2] + Particle2Centroid[2] - Particle1Centroid[2]]
    Particle2Centroid = [Particle2Centroid[0] - Particle1Centroid[0], Particle2Centroid[1] - Particle1Centroid[1], Particle2Centroid[2] - Particle1Centroid[2]]
    # rotate each particle centroid and vectors about the centroid of cube 1
    Particle2Centroid = Particle2Centroid @ inv_1 # x component of orientation vector 
    Particle2VectorXEndPoint = Particle2VectorXEndPoint @ inv_1 # x component of orientation vector 
    Particle2VectorYEndPoint = Particle2VectorYEndPoint @ inv_1 # y component of orientation vector 
    Particle2VectorZEndPoint = Particle2VectorZEndPoint @ inv_1 # z component of orientation vector 
    # shift the particle coordinates back to the original coordinate system
    # Particle2Centroid = Particle2Centroid + Particle1Centroid
    Particle2VectorsX = Particle2VectorXEndPoint - Particle2Centroid
    Particle2VectorsY = Particle2VectorYEndPoint - Particle2Centroid
    Particle2VectorsZ = Particle2VectorZEndPoint - Particle2Centroid
    # normalize
    Particle2VectorsX = Particle2VectorsX / np.linalg.norm(Particle2VectorsX)
    Particle2VectorsY = Particle2VectorsY / np.linalg.norm(Particle2VectorsY)
    Particle2VectorsZ = Particle2VectorsZ / np.linalg.norm(Particle2VectorsZ)
    Particle1Centroid = [0, 0, 0]
    Particle1VectorsX = [1, 0, 0]
    Particle1VectorsY = [0, 1, 0]
    Particle1VectorsZ = [0, 0, 1]
    # vector from Particle1Centroid to Particle2Centroid
    P1P2 = np.array(Particle2Centroid) - np.array(Particle1Centroid)
    P1P2Norm = np.linalg.norm(P1P2)
    AngleBetweenP1P2AndX = calculate_angle_between_vectors(P1P2, [1,0,0])
    AngleBetweenP1P2AndXNegative = calculate_angle_between_vectors(P1P2, [-1,0,0])
    AngleBetweenP1P2AndY = calculate_angle_between_vectors(P1P2, [0,1,0])
    AngleBetweenP1P2AndYNegative = calculate_angle_between_vectors(P1P2, [0,-1,0])
    AngleBetweenP1P2AndZ = calculate_angle_between_vectors(P1P2, [0,0,1])
    AngleBetweenP1P2AndZNegative = calculate_angle_between_vectors(P1P2, [0,0,-1])
    # find out which axis (and direction) has the smallest angle with the P1P2 vector
    MinimumAngle = min(AngleBetweenP1P2AndX,AngleBetweenP1P2AndXNegative,AngleBetweenP1P2AndY,AngleBetweenP1P2AndYNegative,AngleBetweenP1P2AndZ,AngleBetweenP1P2AndZNegative)
    # shift to new coordinate system with cube 1 as center
    Particle2VectorXEndPoint = [Particle2VectorsX[0] + Particle2Centroid[0] - Particle1Centroid[0], Particle2VectorsX[1] + Particle2Centroid[1] - Particle1Centroid[1], Particle2VectorsX[2] + Particle2Centroid[2] - Particle1Centroid[2]]
    Particle2VectorYEndPoint = [Particle2VectorsY[0] + Particle2Centroid[0] - Particle1Centroid[0], Particle2VectorsY[1] + Particle2Centroid[1] - Particle1Centroid[1], Particle2VectorsY[2] + Particle2Centroid[2] - Particle1Centroid[2]]
    Particle2VectorZEndPoint = [Particle2VectorsZ[0] + Particle2Centroid[0] - Particle1Centroid[0], Particle2VectorsZ[1] + Particle2Centroid[1] - Particle1Centroid[1], Particle2VectorsZ[2] + Particle2Centroid[2] - Particle1Centroid[2]]
    Particle2Centroid = [Particle2Centroid[0] - Particle1Centroid[0], Particle2Centroid[1] - Particle1Centroid[1], Particle2Centroid[2] - Particle1Centroid[2]]
    # define the angles to rotate cube 2 about cube 1
    if AngleBetweenP1P2AndX == MinimumAngle:
        VirtualMoveAlpha = np.pi/180 * 0
        VirtualMoveBeta = np.pi/180 *  0
        VirtualMoveGamma = np.pi/180 * 0
    elif AngleBetweenP1P2AndXNegative == MinimumAngle:
        VirtualMoveAlpha = np.pi/180 * 0
        VirtualMoveBeta = np.pi/180 *  0
        VirtualMoveGamma = np.pi/180 * 180
    elif AngleBetweenP1P2AndY == MinimumAngle:
        VirtualMoveAlpha = np.pi/180 * 0
        VirtualMoveBeta = np.pi/180 *  0
        VirtualMoveGamma = np.pi/180 * 90
    elif AngleBetweenP1P2AndYNegative == MinimumAngle:
        VirtualMoveAlpha = np.pi/180 * 0
        VirtualMoveBeta = np.pi/180 *  0
        VirtualMoveGamma = np.pi/180 * -90
    elif AngleBetweenP1P2AndZ == MinimumAngle:
        VirtualMoveAlpha = np.pi/180 * 0
        VirtualMoveBeta = np.pi/180 *  -90
        VirtualMoveGamma = np.pi/180 * 0 
    elif AngleBetweenP1P2AndZNegative == MinimumAngle:
        VirtualMoveAlpha = np.pi/180 * 0
        VirtualMoveBeta = np.pi/180 *  90
        VirtualMoveGamma = np.pi/180 * 0 
    # define the rotation matrices to rotate cube 2 about cube 1
    VirtualMoveRotationX = np.array([[1, 0, 0], [0, math.cos(VirtualMoveAlpha), -math.sin(VirtualMoveAlpha)], [0, math.sin(VirtualMoveAlpha), math.cos(VirtualMoveAlpha)]])
    VirtualMoveRotationY = np.array([[math.cos(VirtualMoveBeta), 0, math.sin(VirtualMoveBeta)], [0, 1, 0], [-math.sin(VirtualMoveBeta), 0, math.cos(VirtualMoveBeta)]])
    VirtualMoveRotationZ = np.array([[math.cos(VirtualMoveGamma), -math.sin(VirtualMoveGamma), 0], [math.sin(VirtualMoveGamma), math.cos(VirtualMoveGamma), 0], [0, 0, 1]])  
    # rotate cube 2 
    Particle2VectorXEndPoint = Particle2VectorXEndPoint @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ
    Particle2VectorYEndPoint = Particle2VectorYEndPoint @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ
    Particle2VectorZEndPoint = Particle2VectorZEndPoint @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ
    Particle2Centroid = Particle2Centroid @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ
    # shift the particle coordinates back to the original coordinate system
    Particle2Centroid = Particle2Centroid + Particle1Centroid
    Particle2VectorsX = Particle2VectorXEndPoint - Particle2Centroid
    Particle2VectorsY = Particle2VectorYEndPoint - Particle2Centroid
    Particle2VectorsZ = Particle2VectorZEndPoint - Particle2Centroid
    # normalize the vectors of cube 2
    Particle2VectorsX = Particle2VectorsX / np.linalg.norm(Particle2VectorsX)
    Particle2VectorsY = Particle2VectorsY / np.linalg.norm(Particle2VectorsY)
    Particle2VectorsZ = Particle2VectorsZ / np.linalg.norm(Particle2VectorsZ)

    return Size1, Size2, Particle1Centroid, Particle2Centroid, Particle1VectorsX, Particle1VectorsY, Particle1VectorsZ, Particle2VectorsX, Particle2VectorsY, Particle2VectorsZ


def is_odd(number):
  if number % 2 == 1:
    return True
  else:
    return False

def findVerticesOfCube(CubeSideLength,COMX,COMY,COMZ,VectorX,VectorY,VectorZ): 
    # Normalizing the Particle2 Vectors
    NormalizedVectorX = VectorX / np.linalg.norm(VectorX)
    NormalizedVectorY = VectorY / np.linalg.norm(VectorY)
    NormalizedVectorZ = VectorZ / np.linalg.norm(VectorZ)
    Centroid = np.array([COMX, COMY, COMZ]) # Angstrom
    CubeSideLength = CubeSideLength # Angstrom
    # Calculating vertices for Particle2 
    Vertex = np.zeros((8, 3))
    # vectorize for faster calculation
    matrix = np.array([[-1,+1, +1],
                      [-1, -1, +1],
                      [-1, -1, -1],
                      [-1, +1, -1],
                      [+1, +1, +1],
                      [+1, -1, +1],
                      [+1, -1, -1],
                      [+1, +1, -1]])
    Vertex = Centroid + np.multiply(CubeSideLength / 2, np.dot(matrix,[NormalizedVectorX, NormalizedVectorY, NormalizedVectorZ]))
    return Vertex

# calculate the position of each bead in a cube
def findAtomPositionsInCube(CubeSize,Sigma,COMX,COMY,COMZ,VectorX,VectorY,VectorZ):
    coordinateList = []
    atomList = []
    if is_odd(CubeSize):
            coordinateList.append(0)
            for i in range(int((CubeSize-1)/2)):
                coordinateList.append((i+1)*Sigma)
                coordinateList.append(-(i+1)*Sigma)
    else: # is even
        for i in range(int(CubeSize/2)):
            coordinateList.append(i*Sigma+Sigma/2)
            coordinateList.append(-(i*Sigma+Sigma/2))
    
    for x in coordinateList:
        for y in coordinateList:
            for z in coordinateList:
                atomList.append([x,y,z])
                
    NormalizedVectorX = VectorX / np.linalg.norm(VectorX)
    NormalizedVectorY = VectorY / np.linalg.norm(VectorY)
    NormalizedVectorZ = VectorZ / np.linalg.norm(VectorZ)
    Centroid = np.array([COMX, COMY, COMZ])          
    matrix = np.array(atomList)
    atomList = np.zeros((len(atomList), 3))
    atomList = Centroid + np.multiply(1, np.dot(matrix,[NormalizedVectorX, NormalizedVectorY, NormalizedVectorZ]))
    atomList = atomList   # the vertices of particle in Angstrom

    return atomList

# calculate the centroid of each vertex in a cube
def findVertexCentroid(CubeSize,Sigma,COMX,COMY,COMZ,VectorX,VectorY,VectorZ):
    coordinateList = []
    atomList = []
    if is_odd(CubeSize):
            coordinateList.append(0)
            for i in range(int((CubeSize-1)/2)-1, int((CubeSize-1)/2)):
                coordinateList.append((i+1)*Sigma)
                coordinateList.append(-(i+1)*Sigma)
    else: # is even
        for i in range(int(CubeSize/2)-1, int(CubeSize/2)):
            coordinateList.append(i*Sigma+Sigma/2)
            coordinateList.append(-(i*Sigma+Sigma/2))
    
    for x in coordinateList:
        for y in coordinateList:
            for z in coordinateList:
                atomList.append([x,y,z])
                
    NormalizedVectorX = VectorX / np.linalg.norm(VectorX)
    NormalizedVectorY = VectorY / np.linalg.norm(VectorY)
    NormalizedVectorZ = VectorZ / np.linalg.norm(VectorZ)
    Centroid = np.array([COMX, COMY, COMZ])          
    matrix = np.array(atomList)
    atomList = np.zeros((len(atomList), 3))
    atomList = Centroid + np.multiply(1, np.dot(matrix,[NormalizedVectorX, NormalizedVectorY, NormalizedVectorZ]))
    atomList = atomList   # the vertices of particle in Angstrom

    return atomList

def CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Cube1X, Cube1Y, Cube1Z, Cube1VectorX,Cube1VectorY,Cube1VectorZ, Cube2X,Cube2Y,Cube2Z, Cube2VectorX,Cube2VectorY,Cube2VectorZ):
    # for some reason, the CheckOverlapForTwoCubes function works more accurately after cubes reoriented to make the first cube aligned with x-, y-, and z-axis
    Size1 = CubeSideLength1 / AASigma
    Size2 = CubeSideLength2 / AASigma
    Particle1Centroid = [Cube1X,Cube1Y,Cube1Z]
    Particle2Centroid = [Cube2X,Cube2Y,Cube2Z]
    Size1, Size2, Particle1Centroid, Particle2Centroid, Cube1VectorX, Cube1VectorY, Cube1VectorZ, Cube2VectorX, Cube2VectorY, Cube2VectorZ = Reorientation(i, j, Size1, Size2, Particle1Centroid, Particle2Centroid, Cube1VectorX, Cube1VectorY, Cube1VectorZ, Cube2VectorX, Cube2VectorY, Cube2VectorZ)
    CubeSideLength1 = Size1 * AASigma
    CubeSideLength2 = Size2 * AASigma
    Cube1X = Particle1Centroid[0]
    Cube1Y = Particle1Centroid[1]
    Cube1Z = Particle1Centroid[2]
    Cube2X = Particle2Centroid[0]
    Cube2Y = Particle2Centroid[1]
    Cube2Z = Particle2Centroid[2]
    
    cube1 = findVerticesOfCube(CubeSideLength1,Cube1X,Cube1Y,Cube1Z,Cube1VectorX,Cube1VectorY,Cube1VectorZ)
    cube2 = findVerticesOfCube(CubeSideLength2,Cube2X,Cube2Y,Cube2Z,Cube2VectorX,Cube2VectorY,Cube2VectorZ)

    for axis in range(3):
        # Calculate the 3 orthogonal axes of the cubes
        axis_vec = np.zeros(3)
        axis_vec[axis] = 1
        # Project the vertices of both cubes onto the axis
        proj1 = [np.dot(vertex, axis_vec) for vertex in cube1]
        proj2 = [np.dot(vertex, axis_vec) for vertex in cube2]
        # Check for overlap on the axis
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False

    # Check for overlap on the remaining 3 face normals of the cubes
    for axis1 in range(3):
        for axis2 in range(3):
            if axis1 != axis2:
                axis_vec = np.zeros(3)
                axis_vec[axis1] = 1
                axis_vec[axis2] = 1

                proj1 = [np.dot(vertex, axis_vec) for vertex in cube1]
                proj2 = [np.dot(vertex, axis_vec) for vertex in cube2]

                if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                    return False

    return True


def calculate_angle_between_vectors(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product / (magnitude_a * magnitude_b))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


def wrapAngle(angle): # wrap angles to between 0-90, this is only true for cubes, other shapes needs different wraping
    if angle <= 90 and angle >= 0:
        angle = angle
    elif angle > 90 and angle <= 180:
        angle = angle - 90
    elif angle > 180 and angle <= 270:
        angle = angle - 180
    elif angle > 270 and angle <= 360:
        angle = angle - 270
    elif angle > -90 and angle < 0:
        angle = 90 + angle 
    elif angle > -180 and angle <= -90:
        angle = 180 + angle 
    elif angle > -270 and angle <= -180:
        angle = 270 + angle
    elif angle > -360 and angle <= -270:
        angle = 360 + angle 
    return angle


def readRestart(restartFile):
    df = pd.read_csv(restartFile)
    Sizes = np.array(list(df['Sizes']))
    Rx = np.array(list(df['Rx']))
    Ry = np.array(list(df['Ry']))
    Rz = np.array(list(df['Rz']))
    VectorXx = list(df['VectorXx'])
    VectorXy = list(df['VectorXy'])
    VectorXz = list(df['VectorXz'])
    VectorYx = list(df['VectorYx'])
    VectorYy = list(df['VectorYy'])
    VectorYz = list(df['VectorYz'])
    VectorZx = list(df['VectorZx'])
    VectorZy = list(df['VectorZy'])
    VectorZz = list(df['VectorZz'])
    VectorX = []
    VectorY = []
    VectorZ = []
    for i in range(len(VectorXx)):
        VectorX.append([VectorXx[i],VectorXy[i],VectorXz[i]])
        VectorY.append([VectorYx[i],VectorYy[i],VectorYz[i]])
        VectorZ.append([VectorZx[i],VectorZy[i],VectorZz[i]])
    VectorX = np.array(VectorX)   
    VectorY = np.array(VectorY)   
    VectorZ = np.array(VectorZ)   
    
    numbers = re.findall(r'\d+\.\d+|\d+', restartFile)
    restartStep = int(numbers[0])  # Convert the first matched number to float
    
    LAMMPSTrajectoryFile = open('LAMMPSTrajectory.lammpstrj', 'a')  # positions in movie format every TrajectoryInterval steps
    EnergyFile = open('Energies.out', 'a')  # stores energies every EnergyOutputInterval steps

    return Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ, restartStep, LAMMPSTrajectoryFile, EnergyFile 


def apply_periodic_boundary(position, BoxLength):
    # if len(position) > 1:
    position = [(p + BoxLength / 2) % BoxLength - BoxLength / 2 for p in position]
    # else:
        # position = (position + BoxLength / 2) % BoxLength - BoxLength / 2 
    return position

    
def calculate_cluster_centroid(particles, BoxLength):
    # pick the central image of the particle positions
    particlesTemp = []
    for particle in particles:
        particle = apply_periodic_boundary(particle, BoxLength)
        particlesTemp.append(particle)
    particles = particlesTemp
    
    # pick the first particle from the cluster 
    # the particle's centroid is considered the temperary centroid of the cluster
    positionOfFirstParticle = particles[0]
    centroid = positionOfFirstParticle # iterative centroid, changes as more particles are considered
    # print('first particle:', centroid)
    # temperary cluster, in which the particle positions are adjusted to next to each other
    # instead of crossing boundaries
    particleTempFirst = centroid.copy() # instead of particleTempFirst=centroid, otherwise particleTempFirst will change with centroid
    clusterTemp = [particleTempFirst]
    for particle in particles:
        if particles.index(particle) != 0:
            particleTemp = []
            for i in range(3):
                distance = particle[i] - centroid[i]
                if distance > BoxLength / 2:
                    distance -= BoxLength
                elif distance < -BoxLength / 2:
                    distance += BoxLength
                particleTemp.append(centroid[i] + distance)
            # the distance of each particle is picked to be the closest to the centroid of the cluster
            # so that they can be easily rotated as a whole
            clusterTemp.append(particleTemp)   
    XSum = 0
    YSum = 0
    ZSum = 0
    for particle in clusterTemp:
        XSum += particle[0]
        YSum += particle[1]
        ZSum += particle[2]
    XAverage = XSum/len(clusterTemp)
    YAverage = YSum/len(clusterTemp)
    ZAverage = ZSum/len(clusterTemp)
    centroid = [XAverage,YAverage,ZAverage]
    
    # pick centroid in the central image
    centroid = apply_periodic_boundary(centroid, BoxLength)
    
    return centroid, clusterTemp


def writeTrajectory(TrajectoryInterval, Style, Step, NumberOfParticles, BoxLengthHalf, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ):
    if Step % TrajectoryInterval == 0:
        if Style == 'Virtual':
            # show as virtual cubes
            with open('LAMMPSTrajectory.lammpstrj', 'a') as LAMMPSTrajectoryFile:
                LAMMPSTrajectoryFile.write('ITEM: TIMESTEP\n')
                LAMMPSTrajectoryFile.write(f'{Step}\n')
                LAMMPSTrajectoryFile.write('ITEM: NUMBER OF ATOMS\n')
                LAMMPSTrajectoryFile.write(f'{NumberOfParticles}\n')
                LAMMPSTrajectoryFile.write('ITEM: BOX BOUNDS pp pp pp\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write('ITEM: ATOMS mol id type x y z radius quatw quati quatj quatk\n')
                for i in range(NumberOfParticles):
                    CubeSideLength = Sizes[i] * AASigma
                    RotationMatrix = np.array([VectorX[i], VectorY[i], VectorZ[i]])
                    # print(RotationMatrix)
                    # Create a Rotation object from the rotation matrix
                    rotation = R.from_matrix(RotationMatrix)
                    # Convert the rotation to a quaternion
                    Quaterion = rotation.as_quat()
                    # Quaterion = R.from_matrix(RotationMatrix).as_quat()
                    LAMMPSTrajectoryFile.write(f'{i} {i} 1 {Rx[i]} {Ry[i]} {Rz[i]} {CubeSideLength/2} {Quaterion[1]} {Quaterion[2]} {Quaterion[3]} {-Quaterion[0]} \n')
        elif Style == 'AA':
            global CGCubeSize, CGSigma
            # Show all atoms in cube
            # print('model',model)
            if model == 'ML':
                CGSigma = AASigma
                # print(CGSigma, AASigma)
                numAtoms = 0
                for size in Sizes:        
                    numAtoms += size**3
            else:
                numAtoms = CGCubeSize**3*NumberOfParticles
            with open('LAMMPSTrajectory.lammpstrj', 'a') as LAMMPSTrajectoryFile:
                LAMMPSTrajectoryFile.write('ITEM: TIMESTEP\n')
                LAMMPSTrajectoryFile.write(f'{Step}\n')
                LAMMPSTrajectoryFile.write('ITEM: NUMBER OF ATOMS\n')
                LAMMPSTrajectoryFile.write(f'{numAtoms}\n')
                LAMMPSTrajectoryFile.write('ITEM: BOX BOUNDS pp pp pp\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write('ITEM: ATOMS mol id type x y z radius\n')
                
                for i in range(NumberOfParticles):
                    if model == 'ML':
                        CGCubeSize = int(Sizes[i])
                    # print('CGCubeSize',CGCubeSize)
                    atomList = findAtomPositionsInCube(CGCubeSize,CGSigma,Rx[i],Ry[i],Rz[i],VectorX[i], VectorY[i], VectorZ[i])
                    NumAtomsPerCube = CGCubeSize**3
                    for j in range(NumAtomsPerCube):
                        atomX = atomList[j][0]
                        atomY = atomList[j][1]
                        atomZ = atomList[j][2]
                        LAMMPSTrajectoryFile.write(f'{i} {i*NumAtomsPerCube + j + 1} 2 {atomX} {atomY} {atomZ} {CGSigma/2} \n')
        elif Style == 'Vertex':
            # Only show vertices atoms in cube
            with open('LAMMPSTrajectory.lammpstrj', 'a') as LAMMPSTrajectoryFile:
                LAMMPSTrajectoryFile.write('ITEM: TIMESTEP\n')
                LAMMPSTrajectoryFile.write(f'{Step}\n')
                LAMMPSTrajectoryFile.write('ITEM: NUMBER OF ATOMS\n')
                # LAMMPSTrajectoryFile.write(f'{NumAtomsPerCube*NumberOfParticles}\n')
                LAMMPSTrajectoryFile.write(f'{8*NumberOfParticles}\n')
                LAMMPSTrajectoryFile.write('ITEM: BOX BOUNDS pp pp pp\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write(f'{-BoxLengthHalf} {BoxLengthHalf}\n')
                LAMMPSTrajectoryFile.write('ITEM: ATOMS mol id type x y z radius\n')
                for i in range(NumberOfParticles):
                    CubeSideLength = Sizes[i] * AASigma
                    atomList = findVerticesOfCube(CubeSideLength,Rx[i],Ry[i],Rz[i],VectorX[i], VectorY[i], VectorZ[i])
                    for j in range(8):
                        atomX = atomList[j][0]
                        atomY = atomList[j][1]
                        atomZ = atomList[j][2]
                        LAMMPSTrajectoryFile.write(f'{i} {i*8 + j + 1} 2 {atomX} {atomY} {atomZ} {AASigma/2} \n')
    return None
    
def writeRestart(RestartFileInterval, lastRestartStep, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ):
    # write restart file
    if Step % RestartFileInterval == 0:
        lastRestartFile = 'restart_'+str(lastRestartStep)+'.csv'
        df = pd.DataFrame()
        df['Sizes'] = Sizes
        df['Rx'] = Rx
        df['Ry'] = Ry
        df['Rz'] = Rz            
        df['VectorXx'] = [row[0] for row in VectorX]
        df['VectorXy'] = [row[1] for row in VectorX]
        df['VectorXz'] = [row[2] for row in VectorX]
        df['VectorYx'] = [row[0] for row in VectorY]
        df['VectorYy'] = [row[1] for row in VectorY]
        df['VectorYz'] = [row[2] for row in VectorY]
        df['VectorZx'] = [row[0] for row in VectorZ]
        df['VectorZy'] = [row[1] for row in VectorZ]
        df['VectorZz'] = [row[2] for row in VectorZ]
        df.to_csv('restart_'+str(Step)+'.csv', index=False)
        if os.path.exists(lastRestartFile):
            os.remove(lastRestartFile)
        lastRestartStep = Step
        
        
    return lastRestartStep

def writeEnergy(EnergyOutputInterval, Step, SystemPotentialEnergy):
    # write out energies every EnergyOutputInterval steps
    if Step % EnergyOutputInterval == 0:
        with open('Energies.out', 'a') as EnergyFile:
            EnergyFile.write(f'{Step} {SystemPotentialEnergy}\n')
        
def writeTime(TimeOutputInterval, Step, Time):
    # write out energies every EnergyOutputInterval steps
    if Step % TimeOutputInterval == 0:
        with open('Time.out', 'a') as TimeFile:
            TimeFile.write(f'{Step} {Time}\n')
            
# PARAMETERS
# AA: All-atom; CG: Coarse-grained
#####################################################################
#################### Shared common input ############################
model = 'AA' # AACG, vdW, ML
visualization = 'Virtual' # AA, Virutal, Vertex
Cores = 10
Step_times = []

kB = 0.0019872  # Boltzmann constant, kcal/(mol*K)
Temperature = 300.0  # Temperature (K)
kBT = kB * Temperature  # kB * Temperature

NumberOfParticles = 1000 # Number of cubes in the system
NPDensity = 1 / 1000000  # System density in cubes/Angstrom^3
AASigma = 2.88 # Angstrom, diameter of one AA bead, i.e., all-atom model sigma

MaxRotation = 30 * np.pi / 180  # maximum allowed rotation for a cube in each trial step
# MaxRotation = 0  # maximum allowed rotation for a cube in each trial step
MaxMove = AASigma  * 12 # maximum allowed translation for a cube in each trial step
EquilibrationSteps = 0  # Equilibration time steps
ProductionSteps = 100  # Production time steps
EnergyOutputInterval = 1  # Energy output intervals
TrajectoryInterval = 1  # Animation output interval
TimeOutputInterval = 1  # Animation output interval
RestartFileInterval = 1000  # Energy output intervals
lastRestartStep = 0 # initialize lastRestartStep
MaxClusterSize = 100 # maximum number of cubes in a cluster. MaxClusterSize = 1: single move; MaxClusterSize > 1: cluster move
#################### Shared common input ############################
#####################################################################

if model == 'AACG':
    ########################################################################
    ############################# AA/CG LJ energy calculation input ##############
    # AA/CG LJ model doesn't allow different sizes of cubes yet
    AAEpsilon = 0.27942 # kcal/mol, AA Epsilon
    CubeSideAASize = 12 # the N in NxNxN atoms in a cube
    CubeSideLength = CubeSideAASize * AASigma # Angstrom, length of the side of cube 1
    CGSigma = 2 * AASigma # Angstrom, diameter of one CG bead
    CubeSideLength = CubeSideAASize * AASigma # Angstrom, length of the side of cube 1
    AACubeSize = round(CubeSideLength / AASigma) # NxNxN beads, using NxNxN beads to represent cube 1
    CGCubeSize = round(CubeSideLength / CGSigma) # NxNxN beads, using NxNxN beads to represent cube 1
    SizeRatioDictionary = {CubeSideAASize: 1} # sizes: NxNxN and their ratios 
    NumAtomsPerCube = CGCubeSize ** 3 # number of beads in each cube, NxNxN
    cutoffAA = 2.5 * AASigma # cutoff distance of AA energy
    cutoffCG = 2.5 * CGSigma # cutoff distance of CG energy
    # key is CGSigma, value is scale factor to match minimum of CG energy with minimum of AA energy
    # these values can be calculated by scaleEpsilonForCG function
    # stroing the scaling factor here is just for skipping the scaling step for efficiency
    ScaleFactorDictionary = {
        1*AASigma: 1,
        2*AASigma: 4.953672744664147,
        3*AASigma: 10.7074512451119,
        4*AASigma: 20.21695243136695,
        6*AASigma: 56.34670620313483,
        12*AASigma: 438.9791464846652 }
    # if the scale factor for a CG model is not found in the ScaleFactorDictionary
    # then call the scaleEpsilonForCG function to calculate the scale factor
    if CGSigma in ScaleFactorDictionary:
        ScaleFactor = ScaleFactorDictionary[CGSigma]
        CGEpsilon = AAEpsilon * ScaleFactor # scale epsilon of AA to match minimum of CG energy with minimum of AA energy
    else:
        ScaleFactor = scaleEpsilonForCG(AASigma,AACubeSize,AAEpsilon,CGSigma,CGCubeSize, cutoffAA, cutoffCG)
        ScaleFactorDictionary[CGSigma] = ScaleFactor
        CGEpsilon = AAEpsilon * ScaleFactor # scale epsilon of AA to match minimum of CG energy with minimum of AA energy
    ############################# AA/CG LJ energy calculation input ##############
    ########################################################################

elif model == 'vdW': 
    ########################################################################
    ############################# vdW energy calculation input ##############
    # vdW model only allows cubes with the same size
    CubeSideAASize = 12 
    SizeRatioDictionary = {CubeSideAASize: 1} # sizes: NxNxN and their ratios 
    CGSigma = 1 * AASigma # vdW is not CG model, this CG size is just for trajectory file outputing AA style purposes 
    CGCubeSize = CubeSideAASize # vdW is not CG model, this CG size is just for trajectory file outputing AA style purposes 
    NumAtomsPerCube = CGCubeSize**3 # vdW is not CG model, this CG size is just for trajectory file outputing AA style purposes 
    # NPSigma = 1.41 * CubeSideLength  # estimated length of a nano particle, Angstrom
    Hamaker = 1.5  # Hamaker constant, (10E^19 J)
    AtomDensity = 0.0585648358351751  # atom density in a cube, atoms/Angstrom^3
    ############################# vdW energy calculation input ##############
    ########################################################################

elif model == 'ML':
    ########################################################################
    ############################# ML energy calculation input ##############
    # ML model allows different sizes of cubes
    # SizeRatioDictionary = {'4': 1, '8': 1, '12': 1} # sizes: NxNxN and their ratios 
    SizeRatioDictionary = {'12': 1,'4': 1} # sizes: NxNxN and their ratios 
    CombinedFilesName = '44To1212'
    # EnergyScaleFactor = -147.069579/(-130.284) # AA_minimum/ML_minimum energy when face-face
    # CubeSideAASize = 12 
    EnergyScaleFactor = 1 # AA_minimum/ML_minimum energy when face-face
    CGSigma = 1 * AASigma # ML is not CG model, this CG size is just for trajectory file outputing AA style purposes 
    # CGCubeSize = CubeSideAASize # ML is not CG model, this CG size is just for trajectory file outputing AA style purposes 
    # NumAtomsPerCube = CGCubeSize**3 # ML is not CG model, this CG size is just for trajectory file outputing AA style purposes 
    # load the saved model from file
    # ML trained models
    with open('../CART-Classification-'+str(CombinedFilesName)+'.pkl', 'rb') as f_ForType:
        MLModel_ForType = pickle.load(f_ForType) 
    # load the saved model from file
    with open('../CART-RegressionType-'+str(CombinedFilesName)+'-0.pkl', 'rb') as f_ForEnergy0:
        MLModel_ForEnergy_Type0 = pickle.load(f_ForEnergy0)
    # load the saved model from file
    with open('../CART-RegressionType-'+str(CombinedFilesName)+'-1.pkl', 'rb') as f_ForEnergy1:
        MLModel_ForEnergy_Type1 = pickle.load(f_ForEnergy1) 
    ############################# ML energy calculation input ##############
    ########################################################################


# The positions are chosen such that there are no major overlaps of particles
# calculate simulation box length according to density and NumberOfParticles
BoxLength = (NumberOfParticles / NPDensity) ** (1.0 / 3.0)  # Angstroms
BoxLengthHalf = BoxLength / 2.0  # Angstroms
# # calculate long range corrections for energy apriori
# ELRC = (1.0 / 3.0) * (NPSigma / CutOff) ** 9 - (NPSigma / CutOff) ** 3
# ELRC = ELRC * (8.0 / 3.0) * NumberOfParticles * np.pi * Epsilon * NPDensity * (NPSigma ** 3)

if __name__ == '__main__':
    totalStart_time = time.time()
    # ################# Initial configuration begins ####################
    # ### two ways to get initial configuration of the cubes:
    # ## Option 1: call the function for initializing positions and orientations
    Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ, restartStep, LAMMPSTrajectoryFile, EnergyFile, TimeFile = InitialConfiguration(NumberOfParticles, BoxLength, AASigma, SizeRatioDictionary)  # Angstrom
    # ## Option 2: extract configuration from a restart file stored from previous simulation
    # Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ, restartStep, LAMMPSTrajectoryFile, EnergyFile = readRestart('restart_23000.csv')
    # ################# Initial configuration ends ####################
    
    # write trajectory file           
    writeTrajectory(TrajectoryInterval,visualization, 0, NumberOfParticles, BoxLengthHalf, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ)
    
    flog = open('log.out', 'w')
    
    print('Starting Initialization')
    print()
    
    SystemEnergyTimeStart = time.time()
    # call the function for calculating total energy of the whole system
    SystemPotentialEnergy = TotalEnergy(Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ)  # Input: Angstrom; output: amu * Angstrom^2 * ps^(-2)
    # SystemPotentialEnergy += ELRC  # amu * A^2 * ps^(-2)
    SystemEnergyTimeEnd = time.time()
    SystemEnergy_time = SystemEnergyTimeEnd - SystemEnergyTimeStart
    print(f"SystemPotentialEnergy Time {SystemEnergy_time:.2f} s") 
    
    Attempt = 0
    Accept = 0
    CountBoltzmann = 0 
    # starting the Monte Carlo Markov chain
    for Step in range(1+restartStep, restartStep+EquilibrationSteps + ProductionSteps + 1):
        StepTimeStart = time.time()
        EarlyTermination = False
        overLapFlag = False
        print('\nStep',Step)
        flog.write(f'\nStep {Step}\n')
        # Equilibration and production stage indicators
        if Step == 1:
            print('\n')
            print('Starting Equilibration')
            # close all figures
            # close all
        if Step == EquilibrationSteps + 1:
            print('\n')
            print('Starting Production')
    
        # write timestep to screen
        if Step % 100 == 0:
            if Step <= EquilibrationSteps:
                print(f'Equilibration Step {Step}')
            else:
                print(f'Production Step {Step}')
                
        #########  Building Cluster Begins ########################
        start_time = time.time()       
        # choose a random particle i as seed
        flog.write('Start Building cluster\n')
        i = int(NumberOfParticles * np.random.uniform(0, 1))
        flog.write(f'current seed {i}\n')
        # position and orientation of the seed
        seedX = Rx[i]  # center of mass in x, Angstrom
        seedY = Ry[i]  # center of mass in y, Angstrom
        seedZ = Rz[i]  # center of mass in z, Angstrom
        seedCentroid = [seedX, seedY, seedZ]
        seedVectorX = VectorX[i] # x component of orientation vector 
        seedVectorY = VectorY[i] # y component of orientation vector
        seedVectorZ = VectorZ[i] # z component of orientation vector
        
        # create a list to recruit particles into a cluster
        clusterParticleList = []
        # recruit the seed particle as the first member of the cluster
        clusterParticleList.append(i)
        # create a list for particles haven't been recruited to the cluster
        ParticlesNotInClusterList = []
        ParticlesNotInClusterList = [ParticleID for ParticleID in range(0, NumberOfParticles)]
        # particles that for sure will be recruited to cluster but due to multiple of such particles bonding with a linker at the same time
        # they will be revisited later after going through one branch
        # each of these particles will begin a new branch of recruiment
        ToBeRecruitedParticlesList = []
        # Temperary List for the j(s) linked with current i
        TemperaryjList = []
        
        # removing the seed particle from ParticlesNotInClusterList
        ParticlesNotInClusterList.remove(i)
        reverseMoveProbabilityList = []
        
        # Virtual move map
        # either translation or rotation, but not at the same time, otherwise bound pairs never address internal angle
        tranOrRot = np.random.uniform(0, 1)
        # tranOrRot = 0.6
        isTranslation = False
        isRotation = False
        if tranOrRot <= 0.5:
            # translation only
            isTranslation = True
            VirtualMoveX = (2 * np.random.uniform(0, 1) - 1) * MaxMove
            VirtualMoveY = (2 * np.random.uniform(0, 1) - 1) * MaxMove
            VirtualMoveZ = (2 * np.random.uniform(0, 1) - 1) * MaxMove
            VirtualMoveAlpha = 0
            VirtualMoveBeta = 0
            VirtualMoveGamma = 0
        else:
            isRotation = True
            # rotation only
            VirtualMoveX = 0
            VirtualMoveY = 0
            VirtualMoveZ = 0   
            # randomAxis = np.random.uniform(0, 1)
            randomAxis = 0.333
            # randomly choose an axis from x,y,z to rotate
            if randomAxis <= 0.333: # pick x-axis to rotate, the other two rotation angles are zero
                VirtualMoveAlpha = (2 * np.random.uniform(0, 1) - 1) * MaxRotation
                VirtualMoveBeta = 0
                VirtualMoveGamma = 0
            elif randomAxis <= 0.666 and randomAxis > 0.333: # pick x-axis to rotate, the other two rotation angles are zero
                VirtualMoveAlpha = 0
                VirtualMoveBeta = (2 * np.random.uniform(0, 1) - 1) * MaxRotation
                VirtualMoveGamma = 0
            else: # pick x-axis to rotate, the other two rotation angles are zero
                VirtualMoveAlpha = 0
                VirtualMoveBeta = 0
                VirtualMoveGamma = (2 * np.random.uniform(0, 1) - 1) * MaxRotation
        VirtualMoveRotationX = np.array([[1, 0, 0], [0, math.cos(VirtualMoveAlpha), -math.sin(VirtualMoveAlpha)], [0, math.sin(VirtualMoveAlpha), math.cos(VirtualMoveAlpha)]])
        VirtualMoveRotationY = np.array([[math.cos(VirtualMoveBeta), 0, math.sin(VirtualMoveBeta)], [0, 1, 0], [-math.sin(VirtualMoveBeta), 0, math.cos(VirtualMoveBeta)]])
        VirtualMoveRotationZ = np.array([[math.cos(VirtualMoveGamma), -math.sin(VirtualMoveGamma), 0], [math.sin(VirtualMoveGamma), math.cos(VirtualMoveGamma), 0], [0, 0, 1]])  
        
        ReverseMoveRotationX = np.array([[1, 0, 0], [0, math.cos(-VirtualMoveAlpha), -math.sin(-VirtualMoveAlpha)], [0, math.sin(-VirtualMoveAlpha), math.cos(-VirtualMoveAlpha)]])
        ReverseMoveRotationY = np.array([[math.cos(-VirtualMoveBeta), 0, math.sin(-VirtualMoveBeta)], [0, 1, 0], [-math.sin(-VirtualMoveBeta), 0, math.cos(-VirtualMoveBeta)]])
        ReverseMoveRotationZ = np.array([[math.cos(-VirtualMoveGamma), -math.sin(-VirtualMoveGamma), 0], [math.sin(-VirtualMoveGamma), math.cos(-VirtualMoveGamma), 0], [0, 0, 1]])
        
        # if maximum cluster size is 1, then there is no need to build cluster
        # because the particle itself is considered a cluster
        if MaxClusterSize > 1:
            Loop = True # continue to loop if Loop is True
        else:
            Loop = False # continue to loop if Loop is True
            overallReverseProbability = 1
        # # to regulate cutoff for cluster size
        Nc = np.random.uniform(0, 1)
        # continue recruiting until cluster size reaches the maximum allowed number
        # or until all particles in the outSideClusterParticleList been checked for recruitment
        ClusterLoopTimeStart = time.time() 
        while Loop is True:
            # check if size of cluster exceeds maximum
            if len(clusterParticleList) >= MaxClusterSize:
                Loop = False                  
            # to introduce cutoff for cluster size
            # to abort the link formation procedure if the cluster size exceeds Nc
            if 1/len(clusterParticleList) < Nc:
                Loop = False # end recruiting particles to cluster
                # EarlyTermination = True
                # print(f'{len(clusterParticleList)} exceed cluster size {1/Nc}!!!')
                break
            Size_i = Sizes[i]
            # Length_i = CubeSideLength # particle side length, Angstrom
            Rx_i_old = Rx[i]  # center of mass in x, Angstrom
            Ry_i_old = Ry[i]  # center of mass in y, Angstrom
            Rz_i_old = Rz[i]  # center of mass in z, Angstrom
            ParticleiCentroid_old = [Rx_i_old, Ry_i_old, Rz_i_old]
            VectorX_i_old = VectorX[i] # x component of orientation vector 
            VectorY_i_old = VectorY[i] # y component of orientation vector
            VectorZ_i_old = VectorZ[i] # z component of orientation vector
            # loop over all j(s) in ParticlesNotInClusterList
            # to find all j(s) that link to the current i
            ParticleNotInClusterTimeStart = time.time()
            # print('ParticlesNotInClusterList',ParticlesNotInClusterList)
            for ParticleNotInCluster in ParticlesNotInClusterList:
                # go through each particle in ParticlesNotInClusterList
                # to check i-j interaction to determine if j should be recruited to cluster
                j = ParticleNotInCluster
                Size_i = Sizes[i]
                Size_j = Sizes[j]
                flog.write(f'current i {i}, size {Size_i}\n')
                flog.write(f'current j {j}, size {Size_j}\n')
                # Length_j = CubeSideLength # particle side length, Angstrom
                Rx_j_old = Rx[j]  # center of mass in x, Angstrom
                Ry_j_old = Ry[j]  # center of mass in y, Angstrom
                Rz_j_old = Rz[j]  # center of mass in z, Angstrom
                ParticlejCentroid_old = [Rx_j_old, Ry_j_old, Rz_j_old]
                VectorX_j_old = VectorX[j] # x component of orientation vector 
                VectorY_j_old = VectorY[j] # y component of orientation vector
                VectorZ_j_old = VectorZ[j] # z component of orientation vector
                # relative distance between i and j
                Rx_ij_old = Rx_i_old - Rx_j_old
                Ry_ij_old = Ry_i_old - Ry_j_old
                Rz_ij_old = Rz_i_old - Rz_j_old
                # smallest relative distance between i and j considering periodic boundary conditions
                Rx_ij_old = Rx_ij_old - BoxLength * round(Rx_ij_old/BoxLength);
                Ry_ij_old = Ry_ij_old - BoxLength * round(Ry_ij_old/BoxLength);
                Rz_ij_old = Rz_ij_old - BoxLength * round(Rz_ij_old/BoxLength);
                # absolute distance between i and j
                ijDistanceSquare_old = Rx_ij_old**2 + Ry_ij_old**2 + Rz_ij_old**2
                ijDistance_old = (ijDistanceSquare_old)**(1/2)
                # i as reference, move j to the image where ij distance is minimal
                Rx_j_old = Rx_i_old - Rx_ij_old
                Ry_j_old = Ry_i_old - Ry_ij_old
                Rz_j_old = Rz_i_old - Rz_ij_old 
                ParticlejCentroid_old = np.array([Rx_j_old, Ry_j_old, Rz_j_old])
                # larger size
                MaxSize = max(Size_i, Size_j)
                CutOff = 2 * MaxSize * AASigma # Angstrom 
                CutOffSquare = CutOff**2  # Angstrom^2
                
                if ijDistanceSquare_old <= CutOffSquare: # check if distance within cutoff distance
                    # Size_i, Size_j, Alpha, Beta, Gamma, X, Y, Z = GetOrientationAndDistance(i, j, Size_i, Size_j, ParticleiCentroid_old, ParticlejCentroid_old, VectorX_i_old, VectorY_i_old, VectorZ_i_old, VectorX_j_old, VectorY_j_old, VectorZ_j_old)
                    # check energy between i and j before vitual move.
                    ijEnergy_old = EnergyBetweenTwoParticles(i, j, Size_i, Size_j, ParticleiCentroid_old, ParticlejCentroid_old, VectorX_i_old, VectorY_i_old, VectorZ_i_old, VectorX_j_old, VectorY_j_old, VectorZ_j_old)  # kcal/mol
                    flog.write(f'ijEnergy_old {ijEnergy_old}\n')
    
                    CubeSideLength1 = Size_i * AASigma
                    CubeSideLength2 = Size_j * AASigma
                    
                    # particle centroid in new coordinate system, with seed particle as the centroid
                    ParticleCentroid = [Rx[i] - seedX, Ry[i] - seedY, Rz[i] - seedZ]
                    ParticleVectorXEndPoint = [VectorX[i][0] + Rx[i] - seedX, VectorX[i][1] + Ry[i] - seedY, VectorX[i][2] + Rz[i] - seedZ]
                    ParticleVectorYEndPoint = [VectorY[i][0] + Rx[i] - seedX, VectorY[i][1] + Ry[i] - seedY, VectorY[i][2] + Rz[i] - seedZ]
                    ParticleVectorZEndPoint = [VectorZ[i][0] + Rx[i] - seedX, VectorZ[i][1] + Ry[i] - seedY, VectorZ[i][2] + Rz[i] - seedZ]
                    
                    # rotate each particle centroid and vectors about the centroid of the seed
                    ParticleCentroid = ParticleCentroid @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ # x component of orientation vector 
                    ParticleVectorXEndPoint = ParticleVectorXEndPoint @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ # x component of orientation vector 
                    ParticleVectorYEndPoint = ParticleVectorYEndPoint @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ # x component of orientation vector 
                    ParticleVectorZEndPoint = ParticleVectorZEndPoint @ VirtualMoveRotationX @ VirtualMoveRotationY @ VirtualMoveRotationZ # x component of orientation vector 
                    
                    # shift the particle coordinates back to the original coordinate system
                    Rx_i_new = ParticleCentroid[0] + seedX
                    Ry_i_new = ParticleCentroid[1] + seedY
                    Rz_i_new = ParticleCentroid[2] + seedZ
                    
                    # new position of i after virtual move
                    Rx_i_new = Rx_i_new + VirtualMoveX  # center of mass in x, Angstrom
                    Ry_i_new = Ry_i_new + VirtualMoveY  # center of mass in y, Angstrom
                    Rz_i_new = Rz_i_new + VirtualMoveZ  # center of mass in z, Angstrom        
                    
                    # apply periodic boundary conditions to positions
                    R_i_new = apply_periodic_boundary([Rx_i_new,Ry_i_new,Rz_i_new], BoxLength)
                    Rx_i_new = R_i_new[0]
                    Ry_i_new = R_i_new[1]
                    Rz_i_new = R_i_new[2]
                    
                    # new vectors of this particle after virtual rotational move
                    VectorX_i_new = [ParticleVectorXEndPoint[0]-ParticleCentroid[0],ParticleVectorXEndPoint[1]-ParticleCentroid[1],ParticleVectorXEndPoint[2]-ParticleCentroid[2]]
                    VectorY_i_new = [ParticleVectorYEndPoint[0]-ParticleCentroid[0],ParticleVectorYEndPoint[1]-ParticleCentroid[1],ParticleVectorYEndPoint[2]-ParticleCentroid[2]]
                    VectorZ_i_new = [ParticleVectorZEndPoint[0]-ParticleCentroid[0],ParticleVectorZEndPoint[1]-ParticleCentroid[1],ParticleVectorZEndPoint[2]-ParticleCentroid[2]]
                    
                    # pickup the central image for the new location of i
                    Rx_i_new = Rx_i_new - BoxLength*round(Rx_i_new/BoxLength);
                    Ry_i_new = Ry_i_new - BoxLength*round(Ry_i_new/BoxLength);
                    Rz_i_new = Rz_i_new - BoxLength*round(Rz_i_new/BoxLength);
    
                    # relatvie distance between new i and old j
                    Rx_ij_new = Rx_i_new - Rx_j_old
                    Ry_ij_new = Ry_i_new - Ry_j_old
                    Rz_ij_new = Rz_i_new - Rz_j_old
                    
                    # smallest relative distance between new i and old j considering periodic boundary conditions
                    Rx_ij_new = Rx_ij_new - BoxLength * round(Rx_ij_new/BoxLength);
                    Ry_ij_new = Ry_ij_new - BoxLength * round(Ry_ij_new/BoxLength);
                    Rz_ij_new = Rz_ij_new - BoxLength * round(Rz_ij_new/BoxLength);
                    
                    # new i as reference, move j to the image where ij distance is minimal
                    Rx_j_old = Rx_i_new - Rx_ij_new
                    Ry_j_old = Ry_i_new - Ry_ij_new
                    Rz_j_old = Rx_i_new - Rz_ij_new
                    
                    ParticleiCentroid_new = [Rx_i_new, Ry_i_new, Rz_i_new]
                    
                    # check energy between i and j after vitual move
                    if CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Rx_i_new,Ry_i_new,Rz_i_new,VectorX_i_new,VectorY_i_new,VectorZ_i_new,Rx_j_old,Ry_j_old,Rz_j_old,VectorX_j_old,VectorY_j_old,VectorZ_j_old):
                        ijEnergy_new = 100 # just a large number
                        overLapFlag = True
                        flog.write(f'{i} and {j} overlap in virtual move\n')
                    else: 
                        # check energy between i and j before vitual move
                        ijEnergy_new = EnergyBetweenTwoParticles(i, j, Size_i, Size_j, ParticleiCentroid_new, ParticlejCentroid_old, VectorX_i_new, VectorY_i_new, VectorZ_i_new, VectorX_j_old, VectorY_j_old, VectorZ_j_old)  # kcal/mol
                        flog.write(f'ijEnergy_new {ijEnergy_new}\n')
                    try: # to get around "OverflowError: math range error" when energy is too large causing exp(Energy) overflow
                        virtualMoveProbability = max(0, 1 - math.exp(ijEnergy_old - ijEnergy_new) / kBT)
                    except OverflowError:
                        virtualMoveProbability = 0
                    flog.write(f'virtualMoveProbability {virtualMoveProbability}\n')
                    # check acceptance probability of this virtual move
                    if virtualMoveProbability >= np.random.uniform(0, 1):
                        # add j to TemperaryjList for later use
                        TemperaryjList.append(j)
                        # calculate reverse move acceptance probability  
                        # particle centroid in new coordinate system, with seed particle as the centroid
                        ParticleCentroid = [Rx[i] - seedX, Ry[i] - seedY, Rz[i] - seedZ]
                        ParticleVectorXEndPoint = [VectorX[i][0] + Rx[i] - seedX, VectorX[i][1] + Ry[i] - seedY, VectorX[i][2] + Rz[i] - seedZ]
                        ParticleVectorYEndPoint = [VectorY[i][0] + Rx[i] - seedX, VectorY[i][1] + Ry[i] - seedY, VectorY[i][2] + Rz[i] - seedZ]
                        ParticleVectorZEndPoint = [VectorZ[i][0] + Rx[i] - seedX, VectorZ[i][1] + Ry[i] - seedY, VectorZ[i][2] + Rz[i] - seedZ]
                        
                        # rotate each particle centroid and vectors about the centroid of the seed
                        ParticleCentroid = ParticleCentroid @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # x component of orientation vector 
                        ParticleVectorXEndPoint = ParticleVectorXEndPoint @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # x component of orientation vector 
                        ParticleVectorYEndPoint = ParticleVectorYEndPoint @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # x component of orientation vector 
                        ParticleVectorZEndPoint = ParticleVectorZEndPoint @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # x component of orientation vector 
                        
                        # shift the particle coordinates back to the original coordinate system
                        Rx_i_reverse = ParticleCentroid[0] + seedX
                        Ry_i_reverse = ParticleCentroid[1] + seedX
                        Rz_i_reverse = ParticleCentroid[2] + seedX
                        
                        # new position of i after reverse move
                        # negative virtual move is reverse move
                        Rx_i_reverse = Rx_i_reverse - VirtualMoveX  # center of mass in x, Angstrom
                        Ry_i_reverse = Ry_i_reverse - VirtualMoveY  # center of mass in y, Angstrom
                        Rz_i_reverse = Rz_i_reverse - VirtualMoveZ  # center of mass in z, Angstrom
                        ParticleiCentroid_reverse = [Rx_i_reverse, Ry_i_reverse, Rz_i_reverse]
                        VectorX_i_reverse = VectorX_i_old @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # x component of orientation vector 
                        VectorY_i_reverse = VectorY_i_old @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # y component of orientation vector
                        VectorZ_i_reverse = VectorZ_i_old @ ReverseMoveRotationX @ ReverseMoveRotationY @ ReverseMoveRotationZ # z component of orientation vector
                        
                        # pickup the central image for the new location of i after reverse move
                        Rx_i_reverse = Rx_i_reverse - BoxLength*round(Rx_i_reverse/BoxLength);
                        Ry_i_reverse = Ry_i_reverse - BoxLength*round(Ry_i_reverse/BoxLength);
                        Rz_i_reverse = Rz_i_reverse - BoxLength*round(Rz_i_reverse/BoxLength);
                        
                        # relative distance between reversed i and old j
                        Rx_ij_reverse = Rx_i_reverse - Rx_j_old
                        Ry_ij_reverse = Ry_i_reverse - Ry_j_old
                        Rz_ij_reverse = Rz_i_reverse - Rz_j_old
                        # smallest relative distance between reversed i and old j considering periodic boundary conditions
                        Rx_ij_reverse = Rx_ij_reverse - BoxLength * round(Rx_ij_reverse/BoxLength);
                        Ry_ij_reverse = Ry_ij_reverse - BoxLength * round(Ry_ij_reverse/BoxLength);
                        Rz_ij_reverse = Rz_ij_reverse - BoxLength * round(Rz_ij_reverse/BoxLength);
                        # reversed i as reference, move j to the image where ij distance is minimal
                        Rx_j_old = Rx_i_reverse - Rx_ij_reverse
                        Ry_j_old = Ry_i_reverse - Ry_ij_reverse
                        Rz_j_old = Rz_i_reverse - Rz_ij_reverse
                        
                        CubeSideLength1 = Sizes[i] * AASigma
                        CubeSideLength2 = Sizes[j] * AASigma
    
                        # check energy between i and j after reverse move
                        if CheckOverlapForTwoCubes(i, j, CubeSideLength1, CubeSideLength2, Rx_i_reverse,Ry_i_reverse,Rz_i_reverse,VectorX_i_reverse,VectorY_i_reverse,VectorZ_i_reverse,Rx_j_old,Ry_j_old,Rz_j_old,VectorX_j_old,VectorY_j_old,VectorZ_j_old):
                            ijEnergy_reverse = 100 # just a large number
                            # overLapFlag = True
                            flog.write(f'{i} and {j} overlap in reverse move\n')
                        else: 
                            ijEnergy_reverse = EnergyBetweenTwoParticles(i, j, Size_i, Size_j, ParticleiCentroid_reverse, ParticlejCentroid_old, VectorX_i_reverse,VectorY_i_reverse,VectorZ_i_reverse, VectorX_j_old, VectorY_j_old, VectorZ_j_old)  # kcal/mol
                            flog.write(f'ijEnergy_reverse {ijEnergy_reverse}\n')
                        # calculate reverse move acceptance probability
                        try: # to get around "OverflowError: math range error" when energy is too large causing exp(Energy) overflow
                            reverseMoveProbability = max(0, 1 - math.exp(ijEnergy_old - ijEnergy_reverse) / kBT)
                        except OverflowError:
                            reverseMoveProbability = 0
                        reverseMoveProbabilityList.append(reverseMoveProbability)
                        flog.write(f'reverseMoveProbabilityList {reverseMoveProbabilityList}\n')
            
            ParticleNotInClusterTimeEnd = time.time()
            ParticleNotInClusterTime = ParticleNotInClusterTimeEnd - ParticleNotInClusterTimeStart
            print(f"ParticleNotInClusterTime {ParticleNotInClusterTime:.2f} s") 
            
            overallReverseProbability = 1
            if len(reverseMoveProbabilityList) > 0:
                for reverseProbability in reverseMoveProbabilityList:
                    overallReverseProbability = overallReverseProbability * reverseProbability
                    flog.write(f'overallReverseProbability {overallReverseProbability}\n')
            if len(TemperaryjList) == 0:
                if len(ToBeRecruitedParticlesList) == 0:
                    Loop = False # end recruiting particles to cluster
                else:
                    # pick a particle from ToBeRecruitedParticlesList as the new i
                    i = random.choice(ToBeRecruitedParticlesList) 
                    # add the new i to clusterParticleList
                    if i not in clusterParticleList: # prevent repeat particles in the list
                        clusterParticleList.append(i)
                        flog.write(f'clusterParticleList {clusterParticleList}\n')
                    # delete the new i from ParticlesNotInClusterList
                    ParticlesNotInClusterList.remove(i)
                    # delete the new i from ToBeRecruitedParticlesList
                    ToBeRecruitedParticlesList.remove(i)
            else: # there are some j(s) that link to i
                # pick a particle from TemperaryjList as new i
                i = random.choice(TemperaryjList)
                # delete the new i from TemperaryjList
                TemperaryjList.remove(i)
                # add the new i to clusterParticleList
                if i not in clusterParticleList: # prevent repeat particles in the list
                    clusterParticleList.append(i)
                    flog.write(f'clusterParticleList {clusterParticleList}\n')
                # delete the new i from ParticlesNotInClusterList
                ParticlesNotInClusterList.remove(i)
                # remove the new i from ToBeRecruitedParticlesList if it exists there
                if i in ToBeRecruitedParticlesList:
                    ToBeRecruitedParticlesList.remove(i)
                # add the rest particles (if any) in TemperaryjList to ToBeRecruitedParticlesList
                if len(TemperaryjList) > 0:
                    for Temperaryj in TemperaryjList:
                        if Temperaryj not in ToBeRecruitedParticlesList:
                            ToBeRecruitedParticlesList.append(Temperaryj)
                # make TemperaryjList empty
                TemperaryjList = []
        ClusterLoopTimeEnd = time.time()
        ClusterLoopTime = ClusterLoopTimeEnd - ClusterLoopTimeStart
        print(f"ClusterLoop {ClusterLoopTime:.2f} s") 
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('ClusterSize: ', len(clusterParticleList))
        print(f"Building cluster:  {elapsed_time:.2f} seconds")         
        # now the cluster is built
        #########  Building Cluster Ends ########################

        if EarlyTermination is False:
            start_time = time.time()
            # calculate energy change after cluster move and before cluster move
            clusterEnergy_old = 0
            clusterEnergy_new = 0
            clusterCentroidXList = []
            clusterCentroidYList = []
            clusterCentroidZList = []
            particlesPositionList = []
            clusterSize = len(clusterParticleList)
            flog.write(f'clusterSize = {clusterSize}\n')
            flog.write(f'clusterParticleList = {clusterParticleList}\n')
            for particleID in clusterParticleList:
                # clusterSize = len(clusterParticleList)
                # print('clusterSize',clusterSize)
                i = particleID
                Size_i = Sizes[i]
                # old position and orientation of this particle before virtual move
                Rx_i_old = Rx[i]  # center of mass in x, Angstrom
                Ry_i_old = Ry[i]  # center of mass in y, Angstrom
                Rz_i_old = Rz[i]  # center of mass in z, Angstrom
                Ri = [Rx_i_old,Ry_i_old,Rz_i_old] # centroid of particle i
                VectorX_i_old = VectorX[i] # x component of orientation vector 
                VectorY_i_old = VectorY[i] # y component of orientation vector
                VectorZ_i_old = VectorZ[i] # z component of orientation vector
                # energy of cluster before cluster move
                # excluding inter-cluster energy to speed up, because we only care about the energy difference before and after cluster move
                currentParticleOldEnergy, overLapFlagOld = OneParticleEnergy(i, Size_i, Rx_i_old, Ry_i_old, Rz_i_old, Sizes, Rx, Ry, Rz, VectorX_i_old, VectorY_i_old, VectorZ_i_old, VectorX, VectorY, VectorZ, NumberOfParticles, clusterParticleList)
                
                clusterEnergy_old = clusterEnergy_old + currentParticleOldEnergy
                flog.write(f'currentParticleOldEnergy {currentParticleOldEnergy}\n')
                particlesPositionList.append([Rx_i_old, Ry_i_old, Rz_i_old])
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Old cluster energy {elapsed_time:.2f} s") 
            flog.write(f'clusterEnergy_old = {clusterEnergy_old}\n')

            
            #######  Virtual Move of Cluster Begins #############
            start_time = time.time()
            # find centroid of the cluster
            centroid, particlesPositionList = calculate_cluster_centroid(particlesPositionList, BoxLength)
            clusterCentroidX = centroid[0]
            clusterCentroidY = centroid[1]
            clusterCentroidZ = centroid[2]
            
            # update particle position to central image
            for particleID in clusterParticleList:
                Rx[particleID] = particlesPositionList[clusterParticleList.index(particleID)][0]
                Ry[particleID] = particlesPositionList[clusterParticleList.index(particleID)][1]
                Rz[particleID] = particlesPositionList[clusterParticleList.index(particleID)][2]
            

            # damping
            if isTranslation is True:
                n_vector = np.array([VirtualMoveX, VirtualMoveY, VirtualMoveZ])
                n_vector = n_vector/np.linalg.norm(n_vector)
                # print('n_vector',n_vector)
                RiMinusRcCrossN_vectorNormSquareList = []
                RadiiOfCubeInCluster = []
                for particleID in clusterParticleList:
                    i = particleID
                    RadiiOfCubeInCluster.append(Sizes[i] * AASigma)
                    # old position and orientation of this particle before virtual move
                    Rx_i_old = Rx[i]  # center of mass in x, Angstrom
                    Ry_i_old = Ry[i]  # center of mass in y, Angstrom
                    Rz_i_old = Rz[i]  # center of mass in z, Angstrom
                    Ri = np.array([Rx_i_old,Ry_i_old,Rz_i_old]) # centroid of particle i
                    Rc = np.array(centroid)
                    # print('Ri-Rc',Ri-Rc)
                    RiMinusRc = Ri-Rc
                    RiMinusRcCrossN_vector = np.cross(RiMinusRc, n_vector) 
                    # print('RiMinusRcCrossN_vector', RiMinusRcCrossN_vector)
                    RiMinusRcCrossN_vectorNorm = np.linalg.norm(RiMinusRcCrossN_vector)
                    # print('RiMinusRcCrossN_vectorNorm', RiMinusRcCrossN_vectorNorm)
                    RiMinusRcCrossN_vectorNormSquare = RiMinusRcCrossN_vectorNorm**2
                    RiMinusRcCrossN_vectorNormSquareList.append(RiMinusRcCrossN_vectorNormSquare)
                AverageRadiusOfCubeInCluster = sum(RadiiOfCubeInCluster)/len(RadiiOfCubeInCluster)
                RiMinusRcCrossN_vectorNormSquareAverage = sum(RiMinusRcCrossN_vectorNormSquareList)/len(RiMinusRcCrossN_vectorNormSquareList)
                EffectiveHydrodynamicRadius = np.sqrt(RiMinusRcCrossN_vectorNormSquareAverage) + AverageRadiusOfCubeInCluster
                # print('EffectiveHydrodynamicRadius',EffectiveHydrodynamicRadius)
                # translational damping factor
                Dt = AverageRadiusOfCubeInCluster/EffectiveHydrodynamicRadius
                Dr = 1
                # print('DrTranslation',Dt)
            elif isRotation is True:
                if VirtualMoveAlpha != 0:
                    n_vector = [1, 0, 0]
                elif VirtualMoveBeta != 0:
                    n_vector = [0, 1, 0]
                elif VirtualMoveGamma != 0:
                    n_vector = [0, 0, 1]   
                else: # if maxrotation is set as 0
                    n_vector = [0, 0, 1] 
                RiMinusRcCrossN_vectorNormSquareList = []
                RadiiOfCubeInCluster = []
                for particleID in clusterParticleList:
                    # print('clusterParticleList', clusterParticleList)
                    i = particleID
                    RadiiOfCubeInCluster.append(Sizes[i] * AASigma)
                    # old position and orientation of this particle before virtual move
                    Rx_i_old = Rx[i]  # center of mass in x, Angstrom
                    Ry_i_old = Ry[i]  # center of mass in y, Angstrom
                    Rz_i_old = Rz[i]  # center of mass in z, Angstrom
                    Ri = np.array([Rx_i_old,Ry_i_old,Rz_i_old]) # centroid of particle i
                    Rc = np.array(centroid)
                    # print('Ri-Rc',Ri-Rc)
                    RiMinusRc = Ri-Rc
                    RiMinusRcCrossN_vector = np.cross(RiMinusRc, n_vector) 
                    # print('RiMinusRcCrossN_vector', RiMinusRcCrossN_vector)
                    RiMinusRcCrossN_vectorNorm = np.linalg.norm(RiMinusRcCrossN_vector)
                    # print('RiMinusRcCrossN_vectorNorm', RiMinusRcCrossN_vectorNorm)
                    RiMinusRcCrossN_vectorNormSquare = RiMinusRcCrossN_vectorNorm**2
                    RiMinusRcCrossN_vectorNormSquareList.append(RiMinusRcCrossN_vectorNormSquare)
                    RiMinusRcCrossN_vectorNormSquareList.append(RiMinusRcCrossN_vectorNormSquare)
                AverageRadiusOfCubeInCluster = sum(RadiiOfCubeInCluster)/len(RadiiOfCubeInCluster)
                RiMinusRcCrossN_vectorNormSquareAverage = sum(RiMinusRcCrossN_vectorNormSquareList)/len(RiMinusRcCrossN_vectorNormSquareList)
                EffectiveHydrodynamicRadius = np.sqrt(RiMinusRcCrossN_vectorNormSquareAverage) + AverageRadiusOfCubeInCluster
                # print('EffectiveHydrodynamicRadius',EffectiveHydrodynamicRadius)
                # rotational damping factor
                Dr = (AverageRadiusOfCubeInCluster/EffectiveHydrodynamicRadius)**3
                Dt = 1
            
            Rx_temp = []
            Ry_temp = []
            Rz_temp = []
            VectorX_temp = []
            VectorY_temp = []
            VectorZ_temp = []
    
            Rx_All_temp = Rx.copy()
            Ry_All_temp = Ry.copy()
            Rz_All_temp = Rz.copy()
            VectorX_All_temp = VectorX.copy()
            VectorY_All_temp = VectorY.copy()
            VectorZ_All_temp = VectorZ.copy()
            
            # rotate each particle about the centroid of the cluster it is in
            for particleID in clusterParticleList:
                # to rotate a particle about the centroid of the cluster
                # is equivalent to shifting the coordinate system to make the centroid of the cluster as the origin
                # of the new coordinate system
                # after performing rotation for each particle about the origin
                # the particles are shifted back to the original coordinate system
                i = particleID
                # particle centroid in new coordinate system
                ParticleCentroid = [Rx[i] - clusterCentroidX, Ry[i] - clusterCentroidY, Rz[i] - clusterCentroidZ]
                ParticleVectorXEndPoint = [VectorX[i][0] + Rx[i] - clusterCentroidX, VectorX[i][1] + Ry[i] - clusterCentroidY, VectorX[i][2] + Rz[i] - clusterCentroidZ]
                ParticleVectorYEndPoint = [VectorY[i][0] + Rx[i] - clusterCentroidX, VectorY[i][1] + Ry[i] - clusterCentroidY, VectorY[i][2] + Rz[i] - clusterCentroidZ]
                ParticleVectorZEndPoint = [VectorZ[i][0] + Rx[i] - clusterCentroidX, VectorZ[i][1] + Ry[i] - clusterCentroidY, VectorZ[i][2] + Rz[i] - clusterCentroidZ]
                
                dampedVirtualMoveRotationX = np.array([[1, 0, 0], [0, math.cos(VirtualMoveAlpha * Dr), -math.sin(VirtualMoveAlpha * Dr)], [0, math.sin(VirtualMoveAlpha * Dr), math.cos(VirtualMoveAlpha * Dr)]])
                dampedVirtualMoveRotationY = np.array([[math.cos(VirtualMoveBeta * Dr), 0, math.sin(VirtualMoveBeta * Dr)], [0, 1, 0], [-math.sin(VirtualMoveBeta * Dr), 0, math.cos(VirtualMoveBeta * Dr)]])
                dampedVirtualMoveRotationZ = np.array([[math.cos(VirtualMoveGamma * Dr), -math.sin(VirtualMoveGamma * Dr), 0], [math.sin(VirtualMoveGamma * Dr), math.cos(VirtualMoveGamma * Dr), 0], [0, 0, 1]])  
                
                # rotate each particle centroid and vectors about the centroid of the cluster
                ParticleCentroid = ParticleCentroid @ dampedVirtualMoveRotationX @ dampedVirtualMoveRotationY @ dampedVirtualMoveRotationZ # x component of orientation vector 
                ParticleVectorXEndPoint = ParticleVectorXEndPoint @ dampedVirtualMoveRotationX @ dampedVirtualMoveRotationY @ dampedVirtualMoveRotationZ # x component of orientation vector 
                ParticleVectorYEndPoint = ParticleVectorYEndPoint @ dampedVirtualMoveRotationX @ dampedVirtualMoveRotationY @ dampedVirtualMoveRotationZ # x component of orientation vector 
                ParticleVectorZEndPoint = ParticleVectorZEndPoint @ dampedVirtualMoveRotationX @ dampedVirtualMoveRotationY @ dampedVirtualMoveRotationZ # x component of orientation vector 
                
                # shift the particle coordinates back to the original coordinate system
                Rx_i_new = ParticleCentroid[0] + clusterCentroidX
                Ry_i_new = ParticleCentroid[1] + clusterCentroidY
                Rz_i_new = ParticleCentroid[2] + clusterCentroidZ
                
                # new position of this particle after virtual translational move
                Rx_i_new = Rx_i_new + VirtualMoveX * Dt  # center of mass in x, Angstrom, transtaltional move suppressed by damping factor Dt
                Ry_i_new = Ry_i_new + VirtualMoveY * Dt  # center of mass in y, Angstrom, transtaltional move suppressed by damping factor Dt
                Rz_i_new = Rz_i_new + VirtualMoveZ * Dt  # center of mass in z, Angstrom, transtaltional move suppressed by damping factor Dt
                
                # new vectors of this particle after virtual rotational move
                VectorX_i_new = [ParticleVectorXEndPoint[0]-ParticleCentroid[0], ParticleVectorXEndPoint[1]-ParticleCentroid[1], ParticleVectorXEndPoint[2]-ParticleCentroid[2]]
                VectorY_i_new = [ParticleVectorYEndPoint[0]-ParticleCentroid[0], ParticleVectorYEndPoint[1]-ParticleCentroid[1], ParticleVectorYEndPoint[2]-ParticleCentroid[2]]
                VectorZ_i_new = [ParticleVectorZEndPoint[0]-ParticleCentroid[0], ParticleVectorZEndPoint[1]-ParticleCentroid[1], ParticleVectorZEndPoint[2]-ParticleCentroid[2]]
                      
                # apply periodic boundary conditions to positions
                R_i_new = apply_periodic_boundary([Rx_i_new,Ry_i_new,Rz_i_new], BoxLength)
                Rx_i_new = R_i_new[0]
                Ry_i_new = R_i_new[1]
                Rz_i_new = R_i_new[2]
                    
                # record the centroid and orientation of each particle
                Rx_temp.append(Rx_i_new)
                Ry_temp.append(Ry_i_new)
                Rz_temp.append(Rz_i_new)
                VectorX_temp.append(VectorX_i_new)
                VectorY_temp.append(VectorY_i_new)
                VectorZ_temp.append(VectorZ_i_new)
                
                Rx_All_temp[i] = Rx_i_new
                Ry_All_temp[i] = Ry_i_new
                Rz_All_temp[i] = Rz_i_new
                VectorX_All_temp[i] = VectorX_i_new
                VectorY_All_temp[i] = VectorY_i_new
                VectorZ_All_temp[i] = VectorZ_i_new
    
            # virtual cluster move finished 
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Virtual move {elapsed_time:.2f} s") 
            
            
            
            clusterOverlapAfterMove = False # at least one particle in the cluster overlap with other particles outside the cluster after cluster move
            start_time = time.time()
            # energy of cluster after cluster move
            for particleID in clusterParticleList:
                i = particleID
                Size_i = Sizes[i]
                Rx_i_new = Rx_All_temp[i]
                Ry_i_new = Ry_All_temp[i]
                Rz_i_new = Rz_All_temp[i]
                VectorX_i_new = VectorX_All_temp[i]
                VectorY_i_new = VectorY_All_temp[i]
                VectorZ_i_new = VectorZ_All_temp[i]
                flog.write(f'clusterParticleList {clusterParticleList}\n')
                flog.write(f'particleID {particleID} in clusterParticleList\n')
                flog.write(f'{i}, {Size_i}, {Rx_i_new}, {Ry_i_new}, {Rz_i_new}, {Sizes}\n')
                # excluding inter-cluster energy to speed up, because we only care about the energy difference before and after cluster move
                currentParticleNewEnergy, overLapFlagNew = OneParticleEnergy(i, Size_i, Rx_i_new, Ry_i_new, Rz_i_new, Sizes, Rx_All_temp, Ry_All_temp, Rz_All_temp, VectorX_i_new, VectorY_i_new, VectorZ_i_new, VectorX_All_temp, VectorY_All_temp, VectorZ_All_temp, NumberOfParticles, clusterParticleList)
                flog.write(f'currentParticleNewEnergy {currentParticleNewEnergy}\n')
                flog.write(f'overLapFlagNew {overLapFlagNew}\n')
                if overLapFlagNew is True:
                    clusterOverlapAfterMove = True
                    flog.write(f'{i} overlap\n')
                clusterEnergy_new = clusterEnergy_new + currentParticleNewEnergy
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"New cluster energy {elapsed_time:.2f} s") 
            
            
            # print('clusterEnergy_new',clusterEnergy_new)
            flog.write(f'clusterEnergy_new = {clusterEnergy_new}\n')
            # # call the function for calculating interaction energy of the chosen particle
            # check for acceptance
            ClusterEnergyChange = clusterEnergy_new - clusterEnergy_old  # delta_V, difference between new and old total energy
            # print('ClusterEnergyChange = ',ClusterEnergyChange)
            Attempt += 1  # the number of attempt
            flog.write(f'ClusterEnergyChange = {ClusterEnergyChange}\n')
           
            # print('clusterParticleList',clusterParticleList)
            RandomNumber = np.random.uniform(0, 1)
            RandomNumberForReverseMoveProbabiliy = np.random.uniform(0, 1)
            try: # to get around "OverflowError: math range error" when energy is too large causing exp(Energy) overflow
                BoltzmannFactor = math.exp(- ClusterEnergyChange / kBT)
            except OverflowError:
                BoltzmannFactor = 0
            if (ClusterEnergyChange <= 0 or (RandomNumber <= BoltzmannFactor and clusterEnergy_new < 0)) and clusterOverlapAfterMove is False:  # accept this move if new total energy is lower than the old one or satisfies Boltzmann factor criteria, make sure no overlap (clusterEnergy_new < 0)            
                if RandomNumberForReverseMoveProbabiliy <= overallReverseProbability:
                    SystemPotentialEnergy += ClusterEnergyChange
                    for particleID in clusterParticleList:
                        i = particleID
                        IndexInCluster = clusterParticleList.index(i)
                        Rx[i] = Rx_temp[IndexInCluster]
                        Ry[i] = Ry_temp[IndexInCluster]
                        Rz[i] = Rz_temp[IndexInCluster]
                        VectorX[i] = VectorX_temp[IndexInCluster]
                        VectorY[i] = VectorY_temp[IndexInCluster]
                        VectorZ[i] = VectorZ_temp[IndexInCluster]
                    Accept += 1
                    # print('Accept cluster Move')
                    flog.write(f'Accept cluster Move!!!\n')
            else:
                flog.write(f'Reject cluster Move~~~\n')
            flog.write(f'SystemPotentialEnergy {SystemPotentialEnergy}\n')
            
            # write trajectory file           
            writeTrajectory(TrajectoryInterval,visualization, Step, NumberOfParticles, BoxLengthHalf, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ)
            
            # write out energies every EnergyOutputInterval steps
            writeEnergy(EnergyOutputInterval, Step, SystemPotentialEnergy)
            
            # write restart file  
            lastRestartStep = writeRestart(RestartFileInterval, lastRestartStep, Sizes, Rx, Ry, Rz, VectorX, VectorY, VectorZ)  
            
        StepEndTime = time.time()
        Step_time = StepEndTime - StepTimeStart
        print(f"Step {Step} {Step_time:.2f} s")         
        writeTime(TimeOutputInterval, Step, Step_time)
        Step_times.append(Step_time)
    print('SystemEnergy_time: ', SystemEnergy_time)
    print('Step_times_sum: ',sum(Step_times))
    flog.close()
    LAMMPSTrajectoryFile.close()
    EnergyFile.close()
    TimeFile.close()
    
    totalEnd_time = time.time()   
    total_Time = totalEnd_time - totalStart_time     
    print(f'total Time cost {total_Time} s')  