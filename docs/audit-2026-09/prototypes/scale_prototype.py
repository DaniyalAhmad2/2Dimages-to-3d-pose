"""H9(a) prototype: per-bone length match (stretch ALONG the bone axis only).

One extra factor per bone, fitted once per take from the subject's median bone
lengths, applied as R @ (I + (k-1) a0 a0^T) so the bone changes length but is
never sheared. Everything else (aim, IK, head) is untouched.
"""
import json, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(os.path.dirname(HERE),"baseline")); sys.path.insert(0,HERE)
import metrics as M
from pose3d.core.skeleton import Joint, NUM_JOINTS, JOINT_NAMES
from pose3d.geometry.character import Character, _DIRECT
from pose3d.geometry.orient import sequence_up, de_tilt_matrix
from pose3d.geometry.bonefit import measure_bone_lengths
u=lambda v:v/(np.linalg.norm(v)+1e-12)

class StretchCharacter(Character):
    def fit_to_subject(self, poses):
        s = super().fit_to_subject(poses)
        poses=np.asarray(poses,float).reshape(-1,NUM_JOINTS,3)
        sub=measure_bone_lengths(poses); rig=self.rig_bone_lengths()
        self._k={}
        for role,(a,b) in _DIRECT.items():
            if role not in self.role: continue
            if not (isinstance(a,Joint) and isinstance(b,Joint)): continue
            key=(int(a),int(b))
            if key in rig and sub.get(key,0)>1e-9 and rig[key]>1e-9:
                self._k[self.role[role]] = float(sub[key]*s/rig[key])
        return s
    def _bone_fk(self, b, base, end):
        M0 = super()._bone_fk(b, base, end)
        k = getattr(self,"_k",{}).get(b)
        if k is None or M0 is base: return M0
        head = base[:3,:3] @ self.head[b] + base[:3,3]
        a0 = u(self.tail[b]-self.head[b])
        S = np.eye(3) + (k-1.0)*np.outer(a0,a0)
        R = M0[:3,:3]
        Mt=np.eye(4); Mt[:3,:3]=R@S; Mt[:3,3]=head-(R@S)@self.head[b]
        return Mt

if __name__=="__main__":
    take=M.load_take("workspace/pose3d_projects/Imported_Session")
    poses=np.asarray(take["fitted3d"],float); R=de_tilt_matrix(sequence_up(poses)); up=poses@R.T
    H=float(np.median([float(p[:,2].max()-p[:,2].min()) for p in up]))
    O={}
    for cls,lbl in ((Character,"shipped (uniform scale only)"),(StretchCharacter,"per-bone length match")):
        ch=cls(); ch.fit_to_subject(up)
        per={}
        for p in up:
            v=~np.isnan(p).any(1); J=ch.posed_joints(p,v,None)
            for j in range(NUM_JOINTS):
                per.setdefault(JOINT_NAMES[j],[]).append(100*float(np.linalg.norm(J[j]-p[j]))/H)
        O[lbl]={"per_joint_pct_height":{k:round(float(np.median(v)),2) for k,v in per.items()},
                "all_median_pct_height":round(float(np.median(np.concatenate([np.asarray(v) for k,v in per.items() if k!="HEAD"]))),2)}
        if lbl.startswith("per-bone"):
            O[lbl]["bone_stretch_factors"]={ch.bone_names[b]:round(k,3) for b,k in ch._k.items()}
    print(json.dumps(O,indent=1))
    json.dump(O,open(os.path.join(HERE,"scale_prototype.json"),"w"),indent=1)
