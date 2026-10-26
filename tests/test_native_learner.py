from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jormungandr.native_learner import NativeBackend, NativeLearnerManager, NativeWorker, NativeWorkerError


def fake_backend(tmp_path: Path, program: str) -> NativeBackend:
    path = tmp_path / 'worker.py'
    path.write_text(program)
    return NativeBackend((sys.executable, str(path)), timeout_seconds=2)


def test_native_calls_are_serialized_and_preserve_response_identity(tmp_path):
    backend = fake_backend(tmp_path, '''import json,sys
counter=0
for line in sys.stdin:
 r=json.loads(line);counter+=1
 print(json.dumps(dict(schema='volt.native_learner.v2',id=r['id'],ok=True,result=dict(counter=counter))),flush=True)
''')
    worker = NativeWorker(backend)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda _: worker.request({'op':'status'}), range(20)))
        assert sorted(r['counter'] for r in replies) == list(range(1,21))
    finally:
        worker.close()
    assert worker.process.poll() is not None


def test_native_timeout_kills_worker_and_prevents_automatic_update_retry(tmp_path):
    backend = fake_backend(tmp_path, 'import time\ntime.sleep(10)\n')
    backend = NativeBackend(backend.command, timeout_seconds=.1)
    worker = NativeWorker(backend)
    with pytest.raises(NativeWorkerError, match='outcome is unknown'):
        worker.request({'op':'update'})
    assert worker.process.poll() is not None
    with pytest.raises(NativeWorkerError, match='unavailable'):
        worker.request({'op':'update'})
    worker.close()


def test_protocol_mismatch_closes_worker(tmp_path):
    backend = fake_backend(tmp_path, '''import json,sys
for line in sys.stdin:
 print(json.dumps(dict(schema='wrong',id=json.loads(line)['id'],ok=True,result={})),flush=True)
''')
    worker = NativeWorker(backend)
    with pytest.raises(NativeWorkerError, match='schema mismatch'):
        worker.request({'op':'status'})
    assert worker.process.poll() is not None


def test_models_accept_only_administrator_registered_executables():
    manager = NativeLearnerManager({})
    with pytest.raises(ValueError, match='not configured'):
        manager.create({'backend':'volt_burn','config':{}})
    with pytest.raises(ValueError, match='unsupported native model fields'):
        manager.create({'command':['arbitrary-executable']})
    manager.close_all()


def test_another_native_engine_can_register_without_volt(tmp_path):
    command = fake_backend(tmp_path, '''import json,sys
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps(dict(schema='example.native.v1',id=r['id'],ok=True,result=dict(version=0,engine='other'))),flush=True)
''').command
    manager = NativeLearnerManager({'other':NativeBackend(command, protocol='example.native.v1')})
    try:
        created = manager.create({'backend':'other','model_id':'independent','config':{}})
        assert created['engine']=='other'
        assert manager.command('independent','status')['version']==0
    finally:
        manager.close_all()


def episode(prefix='train'):
    return {'paths':4,'provenance':{'dataset_id':prefix,'first_path':0},
        'marks':[100,3,101,3.4,102,3.8, 100,3,99,2.6,98,2.2, 100,3,101,3.3,100.5,3.1, 100,3,99,2.7,99.5,2.9],
        'features':[0,0,1,.4, 0,0,-1,-.4, 0,0,1,.3, 0,0,-1,-.3],
        'payoffs':[1,-1,.25,-.25]}


@pytest.mark.skipif(not os.environ.get('VOLT_LEARNER_TEST_BIN'), reason='set VOLT_LEARNER_TEST_BIN to test the real Rust worker')
def test_http_real_burn_matches_direct_updates_and_checkpoint_restart():
    from jormungandr.service import JormungandrHttpServer, JormungandrRuntime
    backend = NativeBackend((os.environ['VOLT_LEARNER_TEST_BIN'],))
    runtime = JormungandrRuntime(native_backends={'volt_burn':backend}, inference_replicas=1)
    server = JormungandrHttpServer('127.0.0.1',0,runtime=runtime)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_address[1]}/v2/native'
    direct=NativeWorker(backend)

    def call(method, route, body=None):
        request=Request(base+route,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'},method=method)
        try:
            with urlopen(request,timeout=10) as response:
                return response.status,json.load(response)
        except HTTPError as e:
            return e.code,json.load(e)

    try:
        assert call('GET','/backends')[1]['backends'][0]['name']=='volt_burn'
        status,created=call('POST','/models',{'model_id':'hedge','backend':'volt_burn','config':{}})
        assert status==200 and created['version']==0
        direct.request({'op':'configure','config':{}})
        registration={'dataset_id':'train','partitions':[{'split':'train','first_path':0,'paths':4}]}
        assert call('POST','/models/hedge/register_dataset',registration)[0]==200
        direct.request({'op':'register_dataset',**registration})
        body={'batch_id':'train','split':'train','batch':episode()}
        assert call('POST','/models/hedge/upload',body)[0]==200
        direct.request({'op':'upload',**body})
        assert call('POST','/models/hedge/update',{'batch_id':'train','expected_version':0})[0]==200
        direct.request({'op':'update','batch_id':'train','expected_version':0})
        status,checkpoint=call('POST','/models/hedge/checkpoint',{})
        assert status==200 and checkpoint=={'ok':True,**direct.request({'op':'checkpoint'})}
        assert call('POST','/models/hedge/update',{'batch_id':'train','expected_version':0})[0]==400
        assert call('POST','/models/hedge/register_dataset',{'dataset_id':'test','partitions':[{'split':'test','first_path':0,'paths':4}]})[0]==200
        assert call('POST','/models/hedge/upload',{'batch_id':'test','split':'test','batch':episode('test')})[0]==200
        assert call('POST','/models/hedge/update',{'batch_id':'test','expected_version':1})[0]==400
        assert call('GET','/models/hedge')[1]['version']==1
        checkpoint=call('POST','/models/hedge/checkpoint',{})[1]
        assert call('POST','/models',{'model_id':'resume','checkpoint':{k:v for k,v in checkpoint.items() if k!='ok'}})[0]==200
        assert call('POST','/models/resume/upload',body)[0]==200
        for name in ['hedge','resume']:
            assert call('POST',f'/models/{name}/update',{'batch_id':'train','expected_version':1})[0]==200
        assert call('POST','/models/hedge/checkpoint',{})[1]==call('POST','/models/resume/checkpoint',{})[1]
        assert call('POST','/models/hedge/infer',{'step':0,'paths':1,'features':[0,0],'previous_holdings':[0,0]})[1]['version']==2
        assert len(call('GET','/models/hedge/metrics')[1]['history'])==2
        assert call('DELETE','/models/resume')[0]==200
    finally:
        direct.close();server.shutdown();thread.join(timeout=3);runtime.close_all();server.server_close()
