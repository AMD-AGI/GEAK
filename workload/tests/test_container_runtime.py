import json
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from container_runtime import runtime_options
import scheduler


def test_runtime_preparation_and_arguments_preserve_spaces(tmp_path):
    script = tmp_path / 'prepare.py'
    seen = tmp_path / 'seen.json'
    script.write_text('import json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n')
    config = tmp_path / 'runtime.json'
    config.write_text(json.dumps({'prepare':[sys.executable,str(script),str(seen),'{container}','a path with spaces'],
        'docker_args':['--user','123:456','-v','/scratch/{container}:/scratch'],
        'entry_prefix':['bash','/entry with spaces.sh']}))
    args,prefix=runtime_options('sample-container',str(config))
    assert json.loads(seen.read_text()) == ['sample-container','a path with spaces']
    assert args[-1] == '/scratch/sample-container:/scratch'
    assert prefix == ['bash','/entry with spaces.sh']


def test_worker_uses_same_runtime_profile_without_changing_device_map(tmp_path,monkeypatch):
    config=tmp_path/'runtime.json'
    config.write_text(json.dumps({'docker_args':['--user','123:456','-v','/local:/local'],
                                 'entry_prefix':['bash','/setup.sh']}))
    monkeypatch.setenv('WORKLOAD_DOCKER_RUNTIME',str(config))
    commands=[]
    monkeypatch.setattr(scheduler,'sh',lambda argv,**kw: (commands.append(argv) or (0,'ok')))
    monkeypatch.setattr(scheduler.os.path,'exists',lambda path: True)
    assert scheduler.DockerBackend(str(tmp_path),'pinned:image').start('musespark025_test',[1],{})
    argv=commands[-1]; image=argv.index('pinned:image')
    assert argv[image+1:image+4] == ['bash','/setup.sh','bash']
    assert '--user' in argv and '/local:/local' in argv
    assert 'GPU_IDS=0' in argv and 'GEAK_GPU_RENDER_DEVICES=renderD136' in argv


@pytest.mark.parametrize('bad',[[],{'docker_args':'--user root'},{'entry_prefix':[None]}])
def test_invalid_runtime_is_rejected(tmp_path,bad):
    config=tmp_path/'runtime.json';config.write_text(json.dumps(bad))
    with pytest.raises(ValueError): runtime_options('owned',str(config))
