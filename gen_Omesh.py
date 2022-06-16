import argparse
import json
import math
from bezier_shapes.shapes_utils import Shape

parser = argparse.ArgumentParser()
parser.add_argument('bezier_cfg')
parser.add_argument('-o', type=str, default='/tmp/', help='output folder')
parser.add_argument('-n', type=int, default=1, help='# of meshes to generate')
parser.add_argument('-v', action='store_true', help='verbose')
parser.add_argument('-m', type=float, default=False, help='quality threshold for conformal mapping calc')
args = parser.parse_args()

cfg = json.load(open(args.bezier_cfg,'r'))

for k in range(args.n):
    success = False
    while not success:
        try:
            if args.m:
                max_retries = 10
                cur_tries = 0
                mapping_quality = (math.inf,math.inf)
                while ((mapping_quality[0] > args.m) or (mapping_quality[1] > args.m)) and (cur_tries<max_retries):
                    shape = Shape(**cfg['shape'])
                    mapping_quality = shape.generate(**cfg['generate'], check_mapping=True)
                    cur_tries += 1
                    if args.v:
                        print(f'Shape {k} - generation attempt {cur_tries}: quality {mapping_quality} | threshold {args.m}')
                if cur_tries == max_retries and ((mapping_quality[0] > args.m) or (mapping_quality[1] > args.m)):
                    raise(RuntimeError('Reached max # of shape generation attempts.'))
            else:
                shape = Shape(**cfg['shape'])
                mapping_quality = shape.generate(**cfg['generate'], check_mapping=False)

            fname, n_el = shape.OMesh(**cfg['mesh'], output_dir=args.o)
            success = True
        except:
            pass
    
    #print(fname)
    #print(f'n elements: {n_el}')
