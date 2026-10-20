import torch
from ref2light.local.masks import patch_mask
from ref2light.runtime.graphs import FrozenCall


class MaskMaker:
    def __init__(self, config, graphs):
        self.config, self.graphs = config, graphs
        self.entries = {}

    def __call__(self, group, crop, full_hw, size, boxes, device):
        if self.config['mode'] != 'context_soft':
            return patch_mask(group,crop,full_hw,size,boxes,device,**self.config)
        values = torch.as_tensor(boxes,dtype=torch.float32)
        if values.ndim!=3 or values.shape[1:]!=(3,4) or not bool(torch.isfinite(values).all()) or not bool((values[...,2:]>values[...,:2]).all()):
            raise ValueError('Facial boxes must be finite K x 3 x 4 rectangles with positive area')
        centres = (values[...,:2]+values[...,2:])/2
        if self.config['split_mode']=='global_median':
            centres = centres.median(dim=0).values[None].expand(values.shape[0],-1,-1)
        eye = centres[:,:2].mean(1)
        dimensions = values.new_tensor([full_hw[1],full_hw[0]])
        axis = (centres[:,2]-eye)*dimensions
        squared = axis.square().sum(-1)
        if not bool((squared>1e-8).all()):
            raise ValueError('Eye and mouth centers coincide; the geometric split is undefined')
        parameters = torch.cat((eye,axis,squared[:,None]),1).to(device)
        crop_tensor = torch.tensor(crop,device=device,dtype=torch.float32)
        key=(group,full_hw,size,values.shape[0],str(device))
        if key not in self.entries:
            h,w=full_hw;ph,pw=size
            x=torch.arange(pw,device=device,dtype=torch.float32)+.5
            y=torch.arange(ph,device=device,dtype=torch.float32)+.5
            ly,lx=torch.meshgrid(y/ph,x/pw,indexing='ij')
            edge=(torch.minimum(torch.minimum(lx,1-lx),torch.minimum(ly,1-ly))/self.config['context_feather']).clamp(0,1)
            context=edge.square()*(3-2*edge)
            dims=dimensions.to(device)
            feather=self.config['split_feather']
            def compute(params, box):
                xx=(box[0]+x*((box[2]-box[0])/pw))/w
                yy=(box[1]+y*((box[3]-box[1])/ph))/h
                yy,xx=torch.meshgrid(yy,xx,indexing='ij')
                xy=torch.stack((xx,yy),-1)
                offset=(xy[None]-params[:,:2,None,None].permute(0,2,3,1))*dims
                projection=(offset*params[:,2:4,None,None].permute(0,2,3,1)).sum(-1)/params[:,4,None,None]
                transition=((projection-(.5-feather/2))/feather).clamp(0,1)
                mouth=transition.square()*(3-2*transition)
                return ((mouth if group=='mouth' else 1-mouth)*context[None])[:,None]
            self.entries[key]=FrozenCall(compute,self.graphs)
        return self.entries[key](parameters,crop_tensor)
