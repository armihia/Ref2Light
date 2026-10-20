import torch
from torch import nn
from torch.nn import functional as F
from ref2light.local.local import LocalControls, LOCAL_FORMAT
from ref2light.local.local_graph import LocalGraphRunner
from ref2light.models.face_adapter import amp

FORMAT = 'companion614_closed_pose_residual'


class ClosedAdapter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input = nn.Conv2d(width+6,32,3,padding=1)
        self.norm = nn.GroupNorm(8,32)
        self.condition = nn.Sequential(nn.Linear(6,32),nn.SiLU(),nn.Linear(32,64))
        self.output = nn.Conv2d(32,5,3,padding=1)
        self.register_buffer('pose_scale',torch.tensor([18.,10.,12.,8.,6.,8.]))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, canonical, reference, raw):
        inputs = torch.cat((features,canonical.to(features.dtype),reference.to(features.dtype)),1)
        x = self.norm(self.input(inputs))
        gamma,beta = self.condition((raw[:,:6]/self.pose_scale).to(inputs.dtype)).chunk(2,-1)
        x = F.silu(x*(1+.25*gamma.tanh()[:,:,None,None])+beta[:,:,None,None])
        return self.output(x)


class PoseLocal(nn.Module):
    """Use a distinct format identifier so incompatible loaders reject the closure head."""
    def __init__(self, config):
        super().__init__()
        self.legacy = LocalControls(config)
        self.closed_adapter = ClosedAdapter(config['width'])

    @property
    def config(self):return self.legacy.config

    @property
    def branches(self):return self.legacy.branches

    def _pose(self, *args):return self.legacy._pose(*args)

    def forward(self, group, canonical, reference, pose, mask=None, identity=None):
        if group!='mouth':
            return self.legacy(group,canonical,reference,pose,mask,identity)
        from ref2light.joint.mouth import extract_mouth_features
        result = extract_mouth_features(self,canonical,reference,pose,mask,identity,zero_rad=False)
        return dict(delta=result['legacy_delta'],visible=result['legacy_visible'],
                    flow=result['flow'],residual=result['residual'],gate=result['gate'])

    def export(self):
        return dict(format=FORMAT,config=dict(self.config),
                    local={k:v.detach().cpu().clone() for k,v in self.state_dict().items()})

    @classmethod
    def from_export(cls,payload,device='cpu'):
        result = cls(payload['config'])
        result.load_state_dict(payload['local'],strict=True)
        return result.to(device)

    @classmethod
    def from_legacy(cls,legacy):
        result = cls(legacy.config)
        result.legacy.load_state_dict(legacy.state_dict(),strict=True)
        return result.to(next(legacy.parameters()).device).eval().requires_grad_(False)


def load_local(payload, device):
    if payload['format']==FORMAT:return PoseLocal.from_export(payload,device)
    if payload['format']==LOCAL_FORMAT:return LocalControls.from_export(payload,device)
    raise ValueError('Unknown local-network format: '+str(payload.get('format')))


class FeatureGraph(LocalGraphRunner):
    def _evaluate(self,group,values):
        from ref2light.joint.mouth import extract_mouth_features
        with amp(values['canonical'].device,True):
            result=extract_mouth_features(self.model,values['canonical'],values['reference'],
                                               values['pose'],values['mask'],values['identity'])
        return {key:result[key] for key in ('features','legacy_visible','legacy_delta')}

    @torch.no_grad()
    def forward(self,canonical,reference,pose,mask=None,identity=None):
        return super().forward('mouth',canonical,reference,pose,mask,identity)

    __call__ = forward


class EagerFeatures:
    """Include the closure residual in the path without CUDA Graphs."""
    graph_count = replays = 0
    def __init__(self,model):self.model=model
    def clear(self):pass
    @torch.no_grad()
    def forward(self,canonical,reference,pose,mask=None,identity=None):
        from ref2light.joint.mouth import extract_mouth_features
        with amp(canonical.device,canonical.device.type=='cuda'):
            result=extract_mouth_features(self.model,canonical,reference,pose,mask,identity)
        return {key:result[key] for key in ('features','legacy_visible','legacy_delta')}

    __call__ = forward
