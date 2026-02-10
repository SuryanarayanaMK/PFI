import numpy as np
genesets_exvivo = {
    1: ['fli1', 'klf1','gata1','gata2','zfpm1','zbtb7a','tal1','spi1'],
    2: ['fli1', 'klf1','gata1','gata2','gfi1b','zbtb7a','runx1','tal1','jun','spi1','zfpm1'],
    3: ['fli1', 'klf1','gata1','gata2','gfi1b','zbtb7a','runx1','tal1','jun','spi1','zfpm1','lmo2','etv6','erg','mef2c'], ### minimal for exvivo
    4: ['fli1', 'klf1','gata1','gata2','gfi1b','zbtb7a','runx1','tal1','jun','spi1','zfpm1','lmo2','etv6','erg','mef2c','cebpa',\
         'nfe2','myc','stat3','nanog','meis1'],
    5: ['fli1', 'klf1','gata1','gata2','gfi1b','zbtb7a','runx1','tal1','jun','spi1','zfpm1','lmo2','etv6','erg','mef2c','cebpa',\
         'nfe2','myc','stat3','nanog','meis1'],
    6: ['fli1', 'klf1','gata1','gata2','gfi1b','zbtb7a','runx1','tal1','jun','spi1','zfpm1','lmo2','etv6','erg','mef2c','cebpa',\
         'nfe2','myc','stat3','nanog','meis1', 'foxo3','hoxa9', 'xbp1','tcf4','ets2', 'ctcf','mllt10','nfia','nfib','myb','mybl2','plek'],
    7: ['gata1','fli1','klf1','gata2','gfi1','gfi1b','runx1','tal1','jun','spi1','zfpm1','lmo2','etv6','erg','cebpa',\
        'meis1','sall4','myc','foxo3','zbtb7a','nanog','nfe2','stat3','mef2c'], # seems complete for exvivo
    8: ['fli1', 'klf1', 'gata1', 'gata2', 'gfi1', 'runx1', 'tal1', 'jun','spi1', 'zfpm1', 'lmo2','etv6','erg','cebpa',\
        'meis1','sall4','myc','foxo3'],
    9: ['fli1', 'klf1', 'gata1', 'gata2', 'gfi1', 'runx1', 'tal1', 'jun','spi1', 'zfpm1', 'lmo2','etv6','erg','cebpa','meis1',\
         'sall4','myc','foxo3','gfi1b','egr1','egr2','nab2'], #### for kaggle specifically 
    10: ['cebpa', 'ctcf', 'egr1', 'egr2', 'erg', 'ets2', 'etv6', 'fli1','foxo3', 'gata1', 'gata2', 'gfi1', 'gfi1b', 'hoxa9', 'jun',
        'klf1', 'lmo2', 'meis1', 'mllt10', 'myb', 'mybl2','myc', 'nab2', 'nfia', 'nfib', 'plek', 'runx1', 'sall4','spi1', 'tcf4', 'tal1', 'xbp1', 'zfpm1'],
    11: ['spi1','zfpm1','fli1','cebpa','gata1','gata2','tal1','klf1','jun','egr1','nab2','gfi1','hoxb4','bmi1','stat3',
        'myc','hoxb6','irf8', 'nfe2','ikzf1','bcl11a','ebf1','pax5','gata3','tcf7','gfi1b','meis1','lyl1','pbx1','erg','etv6','ets1','ets2','tcf3',
        'hoxa9','hoxa5','lmo2','notch1', 'pou2f2', 'runx2','cebpb','myb','mef2c','mllt10','runx1','bcl11b',
        'egr2','tcf4','xbp1','nfib','klf12','mybl2','mxi1','mycn','ctcf','foxo3','foxo1', 'sall4'],
    12: ['spi1','zfpm1','fli1','cebpa','gata1','gata2','tal1','klf1','jun','egr1','nab2','gfi1','hoxb4','bmi1','stat3',
        'myc','hoxb6','irf8', 'nfe2','ikzf1','bcl11a','ebf1','pax5','gata3','tcf7','gfi1b','meis1','lyl1','pbx1','erg','etv6','ets1','ets2','tcf3',
        'hoxa9','hoxa5','lmo2','notch1', 'pou2f2', 'runx2','cebpb','myb','mef2c','mllt10','runx1','bcl11b',
        'egr2','tcf4','xbp1','nfib','klf12','mybl2','mxi1','mycn','ctcf','foxo3','foxo1', 'sall4','cebpe','cebpd','klf4','mafb'],
    13: [
        # Monocyte regulators
        'spi1', 'irf8', 'cebpd', 'klf4', 'mafb',
        'jun', 'egr1', 'nab2',
        # Neutrophil regulators
        'cebpa', 'gfi1', 'cebpb', 'cebpe',
        'stat3', 'myc', 'mef2c', 'runx1', 'myb', 'foxo3',
        # Basophil-related genes (from your list)
        'gata2', 'gata1'
        ],
    14: ['dab2', 'hmgb2', 'gfi1', 'cebpe', 'klf4', 'mxi1', 'cebpb', 'ctnnb1', 'btg1', 'klf6', 'jun', 'zfp36', 'lmo4', 'nme2', 'mxd1', 'tcf7l2', 'mcm7', 'cited2', 'dach1', 'irf8', 'mcm6', 'cebpd', 'arid3a', 'cenpb', 'zeb2', 'sp140', 'creg1', 'mcm5', 'tsc22d4', 'junb', 'dnmt1', 'myb', 'ikzf1', 'rara', 'mcm4', 'zfp36l1', 'e2f8', 'rbbp4', 'mcm2', 'id2', 'erg', 'mcm3', 'nfe2', 'ets1', 'pml', 'rybp', 'pou2f2', 'pwp1', 'suz12', 'irf9']
}

genesets_kaggle_mostvariable = {
    1: ['hbg2',
 'ppbp',
 's100a9',
 'pf4',
 'hbg1',
 's100a8',
 'epx',
 'ccl2',
 'lyz',
 'mt1e',
 'hba2',
 'spp1',
 'gpnmb',
 'prg3',
 'hbb',
 'prtn3',
 'vcan',
 'tpsab1',
 'hba1',
 'ahsp',
 'hpgd',
 'mt1g',
 'cxcl8',
 'mrc1',
 'krt1',
 'mt2a',
 'elane',
 's100b',
 'igfbp5',
 'thbs1',
 'tgfbi',
 'nupr1',
 'tpsb2',
 'cd14',
 'clc',
 'lgals3',
 'ctsb',
 'clec1b',
 'azu1',
 'il7r',
 'lgmn',
 'f13a1',
 'lgalsl',
 'sh3bp5',
 'rnase1',
 'areg',
 'gp9',
 'hbz',
 'tubb1',
 'hla-dqa1',
 'pla2g7',
 'fuca1',
 'ccl5',
 'clec10a',
 'cmtm5',
 'c1qc',
 'hbd',
 'cd9',
 'fam178b',
 'cald1',
 'acp5',
 'ltb',
 'ctsl',
 'alb',
 'hbm',
 'rnase2',
 'mt1x',
 'ms4a2',
 'c15orf48',
 'igsf6',
 'prg2',
 'cyp1b1',
 'gypb',
 'cpa3',
 'ca1',
 'hpse',
 'tcn1',
 'fst',
 'apoe',
 'mpo',
 'hpgds',
 's100a12',
 'plek',
 'treml1',
 'dntt',
 'pkib',
 'cd36',
 'lmo4',
 'tmem40',
 'sgk1',
 'nrgn',
 'il32',
 'cxcl3',
 'hla-drb1',
 'ms4a6a',
 'hla-dqb1',
 'gp1ba',
 'ltbp1',
 'g0s2',
 'srgn']
}


# lx = np.array([
#     4.16,  # fli1
#     2.56,  # klf1
#     2.77,  # gata1
#     4.16,  # gata2
#     3.33,  # gfi1
#     2.77,  # runx1
#     2.31,  # tal1
#     11.1,  # jun
#     3.70,  # spi1
#     2.77,  # zfpm1
#     3.02,  # lmo2
#     3.70,  # etv6
#     2.77,  # erg
#     4.75,  # cebpa
#     2.77,  # meis1
#     4.16,  # sall4
#     13.9,  # myc
#     3.02   # foxo3
# ])


lx_soft = np.array([
    2.2,  # gata1     (↓ from 2.77)
    3.2,  # fli1      (↓ from 4.16)
    2.1,  # klf1      (↓ from 2.56)
    3.2,  # gata2     (↓ from 4.16)
    2.7,  # gfi1      (↓ from 3.33)
    2.4,  # gfi1b     (↓ from 3.02)
    2.2,  # runx1     (↓ from 2.77)
    1.9,  # tal1      (↓ from 2.31)
    8.0,  # jun       (↓ from 11.1)
    3.0,  # spi1      (↓ from 3.70)
    2.2,  # zfpm1     (↓ from 2.77)
    2.4,  # lmo2      (↓ from 3.02)
    3.0,  # etv6      (↓ from 3.70)
    2.2,  # erg       (↓ from 2.77)
    3.6,  # cebpa     (↓ from 4.75)
    2.2,  # meis1     (↓ from 2.77)
    3.2,  # sall4     (↓ from 4.16)
    8.5,  # myc       (↓ from 13.9)
    2.4,  # foxo3     (↓ from 3.02)
    3.0,  # zbtb7a    (↓ from 3.70)
    6.5,  # nanog     (↓ from 8.32)
    2.7,  # nfe2      (↓ from 3.33)
    4.0,  # stat3     (↓ from 5.55)
    3.0   # mef2c     (↓ from 3.70)
])

lx = np.array([
    2.77,  # gata1
    4.16,  # fli1
    2.56,  # klf1
    4.16,  # gata2
    3.33,  # gfi1
    3.02,  # gfi1b
    2.77,  # runx1
    2.31,  # tal1
    11.1,  # jun
    3.70,  # spi1
    2.77,  # zfpm1
    3.02,  # lmo2
    3.70,  # etv6
    2.77,  # erg
    4.75,  # cebpa
    2.77,  # meis1
    4.16,  # sall4
    13.9,  # myc
    3.02,  # foxo3
    3.70,  # zbtb7a
    8.32,  # nanog
    3.33,  # nfe2
    5.55,  # stat3
    3.70   # mef2c
])