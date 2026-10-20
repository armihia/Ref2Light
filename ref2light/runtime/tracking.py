import numpy as np
import torch
from ref2light.local.runtime import pixel_box
from ref2light.runtime.graphs import FrozenCall


class RegionTracker:
    def __init__(self, flow, boxes, state, brow_layout, brow_state, graphs=True):
        self.boxes, self.state = boxes, state
        self.layout, self.brow_state = brow_layout, brow_state
        self.height, self.width = flow.shape[-2:]
        y = 2*(torch.arange(self.height, device=flow.device, dtype=torch.float32)+.5)/self.height-1
        x = 2*(torch.arange(self.width, device=flow.device, dtype=torch.float32)+.5)/self.width-1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xx, yy), -1)
        self.items = []
        items = [(key, value['source_bounds'], value['search_box']) for key,value in state.items()]
        if brow_state is not None:
            items.append(('brows', brow_state['bounds'], brow_state['search']))
        for name, bounds, search in items:
            box = pixel_box(search, self.width, self.height)
            x0,y0,x1,y1 = box
            self.items.append((name, box, bounds, grid[y0:y1,x0:x1],
                torch.arange(y0,y1,device=flow.device), torch.arange(x0,x1,device=flow.device)))
        self.runner = FrozenCall(self.measure, graphs)

    def measure(self, flow):
        values = []
        for _, (x0,y0,x1,y1), bounds, grid, yy, xx in self.items:
            uv = (grid[None]+flow[0,:,:,y0:y1,x0:x1].float().permute(0,2,3,1)).clamp(-1.2,1.2)
            selected = ((uv >= bounds[:,None,None,:2]) & (uv <= bounds[:,None,None,2:])).all(-1).any(0)
            oy, ox = selected.any(1), selected.any(0)
            values.append(torch.stack((torch.where(ox,xx,self.width).amin(),
                torch.where(oy,yy,self.height).amin(),torch.where(ox,xx+1,0).amax(),
                torch.where(oy,yy+1,0).amax(),selected.sum())).float())
        return torch.stack(values) if values else flow.new_empty((0,5))

    @torch.no_grad()
    def __call__(self, flow):
        values = self.runner(flow).cpu().tolist()
        footprints = {item[0]: (v[:4],int(v[4])) for item,v in zip(self.items,values)}
        result, report = {}, {}
        for group in ('eyes','mouth'):
            original = self.boxes[group]
            try:
                if group not in self.state:
                    raise ValueError('Neutral UV calibration is unavailable')
                values, pixels = footprints[group]
                if pixels < 12:
                    raise ValueError('Fewer than 12 visible pixels have matching source UV coordinates')
                current = torch.tensor([values[0]/self.width,values[1]/self.height,
                                        values[2]/self.width,values[3]/self.height])
                neutral = torch.tensor(self.state[group]['footprint'])
                scale = (current[2:]-current[:2])/(neutral[2:]-neutral[:2])
                if not bool(torch.isfinite(scale).all()) or bool(((scale<.35)|(scale>3.)).any()):
                    raise ValueError('The source-UV region scale is out of range')
                initial = torch.tensor(original)
                centre = (initial[:2]+initial[2:])/2+(current[:2]+current[2:]-neutral[:2]-neutral[2:])/2
                radius = (initial[2:]-initial[:2])/2*scale
                box = torch.cat((centre-radius,centre+radius)).clamp(0,1)
                if not bool((box[2:]>box[:2]).all()):
                    raise ValueError('The tracked region no longer has valid area')
                result[group] = box.tolist()
                report[group] = dict(tracked=True,pixels=pixels,scale=scale.tolist())
            except ValueError as exc:
                result[group] = list(original)
                report[group] = dict(tracked=False,fallback='neutral_fixed',error=str(exc))
        if self.boxes.get('view_boxes') is not None:
            initial = torch.as_tensor(self.boxes['view_boxes'],dtype=torch.float64)
            moved = initial.clone()
            for group, indices in (('eyes',(0,1)),('mouth',(2,))):
                old,new = torch.tensor(self.boxes[group],dtype=torch.float64),torch.tensor(result[group],dtype=torch.float64)
                scale = (new[2:]-new[:2])/(old[2:]-old[:2])
                for index in indices:
                    moved[:,index,:2] = new[:2]+(initial[:,index,:2]-old[:2])*scale
                    moved[:,index,2:] = new[:2]+(initial[:,index,2:]-old[:2])*scale
            result['view_boxes'] = moved.tolist()
        layout = self.layout
        if self.brow_state is not None:
            vals,pixels = footprints['brows']
            if pixels < 12:
                raise ValueError('Fewer than 12 visible eyebrow pixels have matching source UV coordinates')
            current = np.asarray(vals)/[self.width,self.height,self.width,self.height]
            old = np.asarray(self.brow_state['footprint'])
            scale = (current[2:]-current[:2])/(old[2:]-old[:2])
            if np.isfinite(scale).all() and not np.any(scale<.35) and not np.any(scale>3):
                boxes = np.asarray(layout['view_boxes'])
                centre = (boxes[...,:2]+boxes[...,2:])/2+(current[:2]+current[2:]-old[:2]-old[2:])/2
                radius = (boxes[...,2:]-boxes[...,:2])/2*scale
                layout = dict(layout,view_boxes=np.concatenate((centre-radius,centre+radius),-1).tolist())
        return result,report,layout

