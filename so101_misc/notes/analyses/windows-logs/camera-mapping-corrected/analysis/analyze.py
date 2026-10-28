from pathlib import Path
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=Path(__file__).resolve().parent
log=root/'EVAL-SO101-2026_09_28-20_27_13--trajectory_check_camera.txt'
feedback=root/'MOTOR-FEEDBACK-2026_09_28-20_27_13--trajectory_check_camera.csv'
text=log.read_text()
records=[(int(m[1]),json.loads(m[2]),json.loads(m[3])) for m in re.finditer(r'step=(\d+) predicted=(\[[^\n]+?\]) safe=(\[[^\n]+?\])',text)]
steps=np.array([r[0] for r in records]); pred=np.array([r[1] for r in records]); limited=np.array([r[2] for r in records])
rows=list(csv.DictReader(feedback.open(encoding='utf-8-sig')))
names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
assert np.array_equal(steps,np.arange(1800))
assert len(rows)==len(steps)*6
for step in steps:
 group=rows[int(step)*6:int(step)*6+6]
 assert {r['motor'] for r in group}==set(names) and all(int(r['step'])==step for r in group)
time=np.array([float(rows[i*6]['timestamp_unix_s']) for i in steps]); time-=time[0]
stats=json.loads((root.parents[2]/'checkpoints/so101-b8-s25000-trimmed-job-1--25000_chkpt/dataset_statistics.json').read_text())['so101_block_into_cup_50_v5']['action']
changed=np.abs(pred-limited)>1e-4
summary={'steps':len(steps),'duration_first_to_last_feedback_seconds':float(time[-1]),'all_predictions_finite':bool(np.isfinite(pred).all()),'all_limited_targets_finite':bool(np.isfinite(limited).all()),'steps_with_any_client_limit':int(changed.any(1).sum()),'median_step_interval_seconds':float(np.median(np.diff(time))),'intervals_over_0_2s':int((np.diff(time)>.2).sum()),'query_time_seconds':float(re.search(r'query_time=([\d.]+)s',text)[1]),'joint_metrics':{},'limitations':['No measured joint positions in these files.','Limited targets are not confirmed final servo commands.','Training min/max are not mechanical joint limits.','Blank errors fields do not establish hardware health.']}
for j,name in enumerate(names):
 rr=[r for r in rows if r['motor']==name]; lo=stats['min'][j]; hi=stats['max'][j]
 summary['joint_metrics'][name]={'predicted_min':float(pred[:,j].min()),'predicted_max':float(pred[:,j].max()),'predicted_first':float(pred[0,j]),'predicted_last':float(pred[-1,j]),'limited_count':int(changed[:,j].sum()),'max_limit_adjustment':float(np.abs(pred[:,j]-limited[:,j]).max()),'training_min':lo,'training_max':hi,'outside_training_range_count':int(((pred[:,j]<lo)|(pred[:,j]>hi)).sum()),'max_abs_current_A':max(abs(float(r['current_A'])) for r in rr),'max_abs_load_pct':max(abs(float(r['load_pct'])) for r in rr),'nonempty_error_fields':sum(bool(r['errors']) for r in rr)}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
fig,axs=plt.subplots(3,2,figsize=(12,9),sharex=True)
for j,ax in enumerate(axs.flat):
 ax.plot(time,pred[:,j],label='Model target',color='#2563eb',linewidth=1.2)
 ax.plot(time,limited[:,j],label='Client-limited target',color='#e67e22',linestyle='--',linewidth=.8)
 idx=changed[:,j]
 if idx.any(): ax.scatter(time[idx],limited[idx,j],s=16,color='#dc2626',zorder=4,label='Limit applied')
 ax.set_title(names[j]); ax.set_ylabel('Recorded gripper units' if j==5 else 'Degrees')
 if j==5: ax.axhline(0,color='gray',linewidth=.7)
 for spine in ('top','right'): ax.spines[spine].set_visible(False)
axs[0,0].legend(loc='best',fontsize=8)
for ax in axs[-1]: ax.set_xlabel('Seconds since first motor-feedback sample')
fig.suptitle('SO-101 trajectory-check: commanded targets, NOT measured robot positions\nCorrected mapping; 1800 steps; 87.49 s; physical tracking cannot be inferred from this plot',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(out/'joint-targets.png',dpi=170); plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='joint_metrics'},indent=2))
print('Long-gap ending-step remainders modulo 30:', sorted(set(int(i+1)%30 for i in np.where(np.diff(time)>.2)[0])))
