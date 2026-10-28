import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
root=Path('/home/yhkim/datasets/rlds/so101_block_into_cup_50_v5/1.1.0')
out=Path('outputs/training-action-audit-20261012')
manifest=json.loads((root/'so101_conversion_manifest.json').read_text())
source={}
for p in Path(manifest['source_root']).glob('data/**/*.parquet'):
 for r in pq.read_table(p,columns=['episode_index','frame_index','action','observation.state']).to_pylist():
  source.setdefault(r['episode_index'],{})[r['frame_index']]=r
meta={r['episode_index']:r for r in manifest['episodes']}
groups={'train':[],'validation':[]}; match=[]
for p in sorted(root.glob('*.tfrecord*')):
 split='validation' if '-validation.' in p.name else 'train'
 for raw in tf.data.TFRecordDataset(str(p)):
  f=tf.train.Example.FromString(raw.numpy()).features.feature
  eid=f['episode_metadata/episode_index'].int64_list.value[0]
  s=np.array(f['steps/observation/state'].float_list.value,dtype=np.float32).reshape(-1,6)
  a=np.array(f['steps/action'].float_list.value,dtype=np.float32).reshape(-1,6)
  start=meta[eid]['effective_first_source_frame']
  src=[r for frame,r in sorted(source[eid].items()) if frame>=start]
  ss=np.array([r['observation.state'] for r in src],dtype=np.float32); aa=np.array([r['action'] for r in src],dtype=np.float32)
  all_src=[r for _,r in sorted(source[eid].items())]
  src_s=np.array([r['observation.state'] for r in all_src],dtype=np.float32)
  src_a=np.array([r['action'] for r in all_src],dtype=np.float32)
  candidates=np.flatnonzero(np.all(src_s==s[0],axis=1) & np.all(src_a==a[0],axis=1))
  exact=[int(j) for j in candidates if j+len(s)<=len(src_s) and np.array_equal(src_s[j:j+len(s)],s) and np.array_equal(src_a[j:j+len(a)],a)]
  match.append({'episode':eid,'length':len(s),'manifest_frame_filter_exact':bool(np.array_equal(s,ss) and np.array_equal(a,aa)), 'source_exact':bool(exact),'source_start_indices':exact, 'source_start_frame_indices':[all_src[j]['frame_index'] for j in exact]})
  groups[split].append((eid,s,a,start))
 print('read',p.name,flush=True)
def describe(d):
 return {'n':len(d),'signed_min':d.min(0).tolist(),'signed_max':d.max(0).tolist(),'abs_p50_p95_p99_max':np.percentile(np.abs(d),[50,95,99,100],axis=0).tolist(),'abs_over_5_count':(np.abs(d)>5).sum(0).tolist()}
report={'joint_order':manifest['joint_names'],'source_alignment':match,'splits':{}}
for split,eps in groups.items():
 report['splits'][split]={'episodes':len(eps),'all_frames':describe(np.concatenate([a-s for _,s,a,_ in eps])),'chunk_valid_frames':describe(np.concatenate([(a-s)[:-29] for _,s,a,_ in eps])),'first_10_frames':describe(np.concatenate([(a-s)[:10] for _,s,a,_ in eps]))}
eps=groups['train']; states=np.concatenate([s[:-29] for _,s,a,_ in eps]); actions=np.concatenate([a[:-29] for _,s,a,_ in eps]); ids=[(eid,start+i) for eid,s,a,start in eps for i in range(len(s)-29)]
stats=json.loads(Path('checkpoints/so101-bidir-b8-s10000--7500_chkpt/dataset_statistics.json').read_text())['so101_block_into_cup_50_v5']['proprio']; scale=np.array(stats['max'])-np.array(stats['min'])
logs=[json.loads(l) for l in Path('outputs/bidir-7500-20261012/actions.jsonl').read_text().splitlines()]
report['real_inputs']={}
for label,r in [('hardware_start',logs[1]),('dryrun_start',logs[4])]:
 q=np.array(r['state']); pred=np.array(r['actions'][0]); dist=np.sqrt(np.mean(((states-q)/scale)**2,axis=1)); ix=np.argsort(dist)[:100]
 report['real_inputs'][label]={'state':q.tolist(),'predicted_first_delta':(pred-q).tolist(),'nearest100_delta_stats':describe((actions-states)[ix]),'nearest5':[{'episode':ids[i][0],'original_frame_from_manifest':ids[i][1],'normalized_rms_distance':float(dist[i]),'state':states[i].tolist(),'delta':(actions[i]-states[i]).tolist()} for i in ix[:5]],'train_count_lift_ge_predicted':int(((actions-states)[:,1]>=pred[1]-q[1]).sum()),'train_count_elbow_le_predicted':int(((actions-states)[:,2]<=pred[2]-q[2]).sum())}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('SUMMARY',json.dumps({'exact_source_match':all(r['source_exact'] for r in match),'splits':report['splits'],'real_inputs':report['real_inputs']},indent=2))
