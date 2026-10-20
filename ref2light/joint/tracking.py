import numpy as np
import torch
from torch.nn import functional as F
from ref2light.local.roi_tracking import source_uv
from ref2light.local.runtime import pixel_box


def _batched_bounds(points):
    """Batch independent quantiles, excluding NaN padding from each region."""
    lengths=[len(value) for value in points]
    maximum=max(lengths)
    packed=torch.stack([F.pad(value,(0,0,0,maximum-len(value)),value=float('nan')) for value in points])
    result=torch.nanquantile(packed,packed.new_tensor([.2,.8]),dim=1)
    actual=torch.arange(maximum,device=packed.device)[None]<torch.tensor(lengths,device=packed.device)[:,None]
    invalid=(~torch.isfinite(packed).all(-1)&actual).any(-1)
    result=result.masked_fill(invalid[None,:,None],float('nan'))
    return torch.cat((result[0],result[1]),-1)


@torch.no_grad()
def _centres(flow,state):
    height,width=flow.shape[-2:]
    x0,y0,x1,y1=state['crop']
    bounds=state['bounds'][:,:,None,None]
    yy=(torch.arange(y0,y1,device=flow.device,dtype=torch.float32)+.5)/height
    xx=(torch.arange(x0,x1,device=flow.device,dtype=torch.float32)+.5)/width
    yy,xx=torch.meshgrid(yy,xx,indexing='ij')
    xy=torch.stack((xx,yy),-1)[None,None]
    # Read only the face search area when tracking local regions.
    field=(xy*2-1+flow[0,:, :,y0:y1,x0:x1].float().permute(0,2,3,1)[:,None]).clamp(-1.2,1.2)
    search=state['search'][:,:,None,None]
    selected=((field>=bounds[...,:2])&(field<=bounds[...,2:])).all(-1)
    selected &= ((xy>=search[...,:2])&(xy<=search[...,2:])).all(-1)
    count=selected.sum((-2,-1))
    x=(selected*xx).sum((-2,-1))/count.clamp_min(1)
    y=(selected*yy).sum((-2,-1))/count.clamp_min(1)
    # Transfer all views together to avoid per-region GPU synchronization.
    return torch.stack((x,y,count.float()),-1).cpu().numpy()


@torch.no_grad()
def prepare_core_tracking(flow,boxes):
    views=np.asarray(boxes['view_boxes'],dtype=np.float32)
    uv=source_uv(flow)
    if views.shape!=(uv.shape[0],3,4):raise ValueError('The number of per-view facial boxes does not match the anchor count')
    h,w=flow.shape[-2:]
    points=[]
    for k in range(len(views)):
        for box in views[k]:
            x0,y0,x1,y1=pixel_box(box,w,h)
            points.append(uv[k,y0:y1,x0:x1].reshape(-1,2))
    bounds=_batched_bounds(points).reshape(len(views),3,4)
    centre=(views[...,:2]+views[...,2:])/2
    radius=(views[...,2:]-views[...,:2])*2
    # Set the search extent from the detected facial region size.
    search=np.clip(np.concatenate((centre-radius,centre+radius),-1),0,1)
    outer=np.r_[search[...,:2].reshape(-1,2).min(0),search[...,2:].reshape(-1,2).max(0)]
    state=dict(bounds=bounds,search=torch.tensor(search,device=flow.device),
               crop=pixel_box(outer,w,h),views=views,boxes=boxes)
    state['neutral']=_centres(flow,state)
    return state


@torch.no_grad()
def track_core_boxes(flow,boxes,state):
    current=_centres(flow,state)
    valid=(current[...,2]>=12)&(state['neutral'][...,2]>=12)
    displacement=np.where(valid[...,None],current[...,:2]-state['neutral'][...,:2],0.)
    views=state['views']+np.concatenate((displacement,displacement),-1)
    result=dict(view_boxes=views.tolist())
    for name,indices in (('eyes',(0,1)),('mouth',(2,))):
        chosen=displacement[:,indices].reshape(-1,2)
        good=valid[:,indices].reshape(-1)
        delta=np.median(chosen[good],axis=0) if good.any() else np.zeros(2)
        original=np.asarray(boxes[name],dtype=np.float32)
        result[name]=np.clip(original+np.r_[delta,delta],0,1).tolist()
    return result,dict(mode='landmark_core_uv',tracked_features=int(valid.sum()),
                       total_features=int(valid.size),displacements=displacement.tolist())
