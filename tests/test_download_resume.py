"""A retry must skip replayed source images rather than inflate the quota counts."""
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
from src.data import subsample_sid_streaming


def test_stream_retry_keeps_distinct_images(tmp_path):
    rows=[{'label':label,'img_id':str(i),'image':Image.fromarray(np.full((16,16,3),i*40,dtype=np.uint8))}
          for i,label in enumerate([0,0,1,2],start=1)]
    calls={'train':0,'validation':0}
    def load(*a,split,**kwargs):
        calls[split]+=1
        if split=='train' and calls[split]==1:
            def interrupted():
                yield rows[0]
                raise ConnectionError('simulated interrupted download')
            return interrupted()
        return iter(rows)
    with patch('datasets.load_dataset',side_effect=load),patch('src.data._resolve_hf_token',return_value=None),patch('time.sleep'):
        result=subsample_sid_streaming(tmp_path,2,2)
    reals=list((tmp_path/'train/real').glob('*.jpg'))
    assert len(reals)==2 and len({p.read_bytes() for p in reals})==2
    assert result['unique_file_hashes']['train']==4
    assert calls['train']==2
