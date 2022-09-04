def toPAD(occ):
    angry = occ[0]
    disgust = occ[1]
    fear = occ[2]
    happy = occ[3]
    sad = occ[4]
    surprise = occ[5]

    P = 0
    A = 0
    D = 0

    P += angry * -0.51
    A += angry * 0.59
    D += angry * 0.25

    P += disgust * -0.4
    A += disgust * 0.2
    D += disgust * 0.1

    P += fear * -0.64
    A += fear * 0.60
    D += fear * -0.43

    P += happy * 0.40
    A += happy * 0.20
    D += happy * 0.15

    P += sad * -0.40
    A += sad * -0.20
    D += sad * -0.50

    P += surprise * 0.20
    A += surprise * 0.45
    D += surprise * -0.45

    pad = [P, A, D]
    return pad

def toDTASL(pad):
    P = pad[0]
    A = pad[1]
    D = pad[2]

    Depression =   -.42*P + .09*A - .37*D
    TraitAnxiety = -.43*P + .29*A - .37*D
    Loneliness =   -.77*P + .00*A - .23*D

    return [Depression, TraitAnxiety, 0, 0, Loneliness, 0]
