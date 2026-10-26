"""Two scenario-producing HTTP actors train one resident Volt/Burn policy.

Start jormungandr-service with VOLT_LEARNER_BIN pointing to the compiled Volt
native_learner, then run this example. All market scenarios here are synthetic.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
from pathlib import Path
import random
import time
import uuid

from jormungandr.client import JormungandrClient, JormungandrClientError


def call_price(spot, maturity):
    sigma, rate, dividend = .2, .03, .01
    d1=(math.log(spot/100)+(rate-dividend+.5*sigma*sigma)*maturity)/(sigma*math.sqrt(maturity))
    d2=d1-sigma*math.sqrt(maturity)
    normal=lambda x:.5*(1+math.erf(x/math.sqrt(2)))
    return spot*math.exp(-dividend*maturity)*normal(d1)-100*math.exp(-rate*maturity)*normal(d2)


def scenario_batch(paths, prefix, seed):
    rng=random.Random(seed)
    batch={'paths':paths,'provenance':{'dataset_id':f'{prefix}-seed-{seed}','first_path':0},'marks':[],'features':[],'payoffs':[]}
    for _ in range(paths):
        spots=[100.]
        for _ in range(2):
            spots.append(spots[-1]*math.exp((.03-.01-.5*.2**2)/252+.2/math.sqrt(252)*rng.gauss(0,1)))
        for t,spot in enumerate(spots):
            batch['marks'].extend([spot,call_price(spot,(84-t)/252)])
            if t<2:
                batch['features'].extend([math.log(spot/100)/.02,(2-t)/2])
        batch['payoffs'].append(abs(spots[-1]-100))
    return batch


def run(url, *, paths=64, updates=20, checkpoint=None, backend='volt_burn', target='cpu', precision=None):
    client=JormungandrClient(url,timeout=65)
    model_id='native-hedge-'+uuid.uuid4().hex[:12]
    call=call_price(100,2/252)
    premium=2*call-100*math.exp(-.01*2/252)+100*math.exp(-.03*2/252)
    config={'target':{'cuda':{'index':0}} if target=='cuda' else target,
            'precision':precision or ('f32' if target=='metal' else 'f64'),
            'premium':premium,'trading_costs':[.0005,.01],'liquidation_costs':[.0005,.01],
            'cash_growth':[math.exp(.03/252)]*2,'income_growth':[math.exp(.01/252),1.]*2}
    created=client.create_native_model({'model_id':model_id,'backend':backend,'config':config})
    try:
        validation_batch=scenario_batch(paths,'validation',999)
        client.native_command(model_id,'register_dataset',{'dataset_id':validation_batch['provenance']['dataset_id'],'partitions':[{'split':'validation','first_path':0,'paths':paths}]})
        client.native_command(model_id,'upload',{'batch_id':'validation','split':'validation','batch':validation_batch})
        initial=client.native_command(model_id,'evaluate',{'batch_id':'validation'})

        def actor(index):
            remote=JormungandrClient(url,timeout=65)
            batch_id=f'actor-{index}'
            batch=scenario_batch(paths,batch_id,123+index)
            remote.native_command(model_id,'register_dataset',{'dataset_id':batch['provenance']['dataset_id'],'partitions':[{'split':'train','first_path':0,'paths':paths}]})
            remote.native_command(model_id,'upload',{'batch_id':batch_id,'split':'train','batch':batch})
            retries=0
            for _ in range(updates):
                while True:
                    version=remote.get_native_model(model_id)['version']
                    try:
                        remote.native_command(model_id,'update',{'batch_id':batch_id,'expected_version':version})
                        break
                    except JormungandrClientError as exc:
                        if 'stale expected_version' not in str(exc):
                            raise
                        retries+=1
            return {'actor':index,'updates':updates,'version_conflicts':retries}

        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            actors=list(pool.map(actor,[0,1]))
        elapsed=time.perf_counter()-start
        final=client.native_command(model_id,'evaluate',{'batch_id':'validation'})
        state=client.native_command(model_id,'checkpoint')
        if checkpoint is not None:
            checkpoint=Path(checkpoint)
            temporary=checkpoint.with_suffix(checkpoint.suffix+'.tmp')
            temporary.write_text(json.dumps({k:v for k,v in state.items() if k!='ok'})+'\n')
            temporary.replace(checkpoint)
        return {'source':'synthetic constant-volatility European fixture','backend':created['backend'],
                'policy_engine':created['policy_engine'],'precision':created['precision'],'target':created['target'],
                'fusion_active':created['fusion_active'],'actors':actors,
                'version':final['version'],'initial_validation_loss':initial['loss'],'final_validation_loss':final['loss'],
                'updates_per_second_including_http':2*updates/elapsed,'elapsed_seconds':elapsed,
                'paths_per_actor_batch':paths,'checkpoint_saved':str(checkpoint) if checkpoint else None}
    finally:
        client.delete_native_model(model_id)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--paths',type=int,default=64)
    parser.add_argument('--updates',type=int,default=20,help='updates per actor')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--backend',default='volt_burn')
    parser.add_argument('--target',choices=['cpu','metal','cuda'],default='cpu')
    parser.add_argument('--precision',choices=['f32','f64'])
    args=parser.parse_args()
    if not 1<=args.paths<=512 or not 1<=args.updates<=10000:
        parser.error('paths must be 1..512; updates must be 1..10000')
    print(json.dumps(run(args.url,paths=args.paths,updates=args.updates,checkpoint=args.checkpoint,backend=args.backend,target=args.target,precision=args.precision),indent=2))
