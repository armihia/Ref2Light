import gc
from pathlib import Path
import torch
from torch.nn import functional as F
from ref2light.common.controls import validate_pose
from ref2light.models.face_adapter import amp
from ref2light.common.io import read_json,write_json
from ref2light.local.runtime import canonical_pose,pixel_box,soft_mask
from ref2light.local.inference_speed import branch_is_inactive
from ref2light.brows.regions import (BrowDetector,prepare_brow_tracking,validate_layout,crop_bounds,support_mask)
from ref2light.joint.pose_mouth import FeatureGraph,EagerFeatures
from ref2light.joint.mouth import extract_mouth_features
from ref2light.joint.tracking import prepare_core_tracking,track_core_boxes
from .reference_session import Session as ReferenceSession,ReferenceLocalizationError
from .tracking import RegionTracker
from .masks import MaskMaker
from .graphs import FrozenCall


class Session(ReferenceSession):
    def __init__(self,*args,**kwargs):
        deferred = kwargs.pop('defer_reference',False)
        self.fast,self.mask_maker = True,None
        self._core_tracking = None
        self._brow_layout = self._brow_tracking = None
        self._brow_error = None
        self._fast_trackers = {}
        super().__init__(*args,defer_reference=True,**kwargs)
        graphs = self.local_graphs_requested and self.device.type=='cuda'
        self.mask_maker = MaskMaker(self.mask_config,graphs)
        self.brow_runner = FrozenCall(self._brow_forward,graphs)
        self.rad_runner = FrozenCall(self._rad_forward,graphs)
        self.rad_feature_graph = FeatureGraph(self.local) if graphs else EagerFeatures(self.local)
        if not deferred:self.set_reference(args[1] if len(args)>1 else kwargs['reference'])

    def _clear_cache(self):
        super()._clear_cache()
        self._fast_frame_key = self._fast_frame_value = None
        self._fast_trackers = {}
        self._tracking_includes_brows = False
        gc.collect()

    @torch.no_grad()
    def set_reference(self,*args,**kwargs):
        previous = self._core_tracking
        self._core_tracking = None
        try:
            super().set_reference(*args,**kwargs)
        except BaseException:
            self._core_tracking = previous
            raise
        self._brow_layout = self._brow_tracking = None
        self._brow_error = None
        from PIL import Image
        array = self.ref[0].detach().float().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
        detector = BrowDetector(self.landmark_model)
        try:
            self._brow_layout = detector.locate(self._context_value['anchors'],Image.fromarray(array),self.raw_boxes)
            self._brow_layout['reference'] = self.reference
            self._brow_tracking = prepare_brow_tracking(self._neutral_flow,self._brow_layout) if self.roi_tracking=='source_uv' else None
            write_json(self.folder/'brow_regions.json',self._brow_layout)
        except ValueError as exc:
            self._brow_error = str(exc)
            print('Eyebrow localization failed; mark eyebrow regions in the controller: ',exc,flush=True)
        finally:detector.close()
        if self.boxes.get('view_boxes') is not None:
            self._core_tracking = prepare_core_tracking(self._neutral_flow,self.boxes)
        self._clear_cache()

    def set_output_boxes(self,path):
        super().set_output_boxes(path)
        self._core_tracking = (prepare_core_tracking(self._neutral_flow,self.boxes)
                               if self.boxes.get('view_boxes') is not None else None)
        self._clear_cache()

    def _updated_context(self,raw,brows):
        canonical = list(raw)
        if canonical[8]>0:canonical[8]=0.
        if brows:canonical[9:12]=[0.,0.,0.]
        neutral = tuple(canonical_pose(canonical,self.eye_mode))
        need_brows = brows and self._brow_layout is not None
        if neutral!=self._context_key or (need_brows and not self._tracking_includes_brows):
            anchors,aux = self._anchors(self.ref,self.features,neutral)
            if self._core_tracking is not None:
                boxes,report = track_core_boxes(aux['flow'],self.boxes,self._core_tracking)
            else:boxes,report = self.boxes,dict(mode='neutral_fixed')
            layout = self._brow_layout
            if need_brows and self.roi_tracking=='source_uv' and self._brow_tracking is not None:
                if True not in self._fast_trackers:
                    self._fast_trackers[True]=RegionTracker(aux['flow'],self.boxes,{},
                        self._brow_layout,self._brow_tracking,self.local_graphs_requested)
                _,_,layout=self._fast_trackers[True](aux['flow'])
            self.current_boxes,self.current_tracking_report=boxes,report
            self._tracked_brow_layout=layout
            self._tracking_includes_brows=need_brows
            self._context_value=self._make_context(anchors,aux,self.features,boxes)
            self._context_key=neutral
        return self._context_value

    def metadata(self):
        from ref2light import __version__
        return dict(version=__version__,checkpoint=str(self.model_path),training_step=self.payload['step'],
            reference=self.reference,reference_preparation=self.reference_report,native_size_wh=list(self.size),
            output_boxes=self.current_boxes,brow_regions=self._brow_layout,brow_detection_error=self._brow_error,
            runtime_config=self.resolved_runtime_config,mouth_tracking='core',model_choice='complete',
            compute_device=str(self.device),local_graphs=self.local_graph_runner is not None,
            body_graphs=self.body_graph_runner is not None,
            teacher_used_at_inference=False,geometry_input_used=False,target_Y_input_used=False)

    def set_brow_boxes(self, path):
        value = read_json(path)
        if value.get('alignment_enabled', self.align) != self.align:
            raise ValueError('The eyebrow boxes use a different alignment setting; reannotate them')
        self._brow_layout = validate_layout(value, self._neutral_flow.shape[1], self.reference)
        self._brow_tracking = prepare_brow_tracking(self._neutral_flow, self._brow_layout) if self.roi_tracking == 'source_uv' else None
        self._brow_error = None
        self._clear_cache()


    def manual_brow_request(self):
        request = self.manual_reference_request()
        request['reason'] = 'manual_brow_adjustment'
        return request


    def manual_reference_request(self):
        result = super().manual_reference_request()
        # Preview generation replays the body graph, so rebuild source-texture context on the next frame.
        self._context_key = None
        return result


    def _make_context(self, anchors, aux, features, boxes=None):
        if not self.fast or self.mask_maker is None:
            return super()._make_context(anchors, aux, features, boxes)
        boxes = self.boxes if boxes is None else boxes
        # Graph buffers remain valid only until the next body replay; copy base colours retained across frames.
        self._brow_warped, self._current_brow_flow = aux['warped'], aux['flow']
        anchors = anchors.clone() if self.body_graph_runner is not None else anchors
        result = dict(anchors=anchors, identity=features[-1].float().mean((2,3)), patches={})
        for group,size in (('eyes',(96,192)),('mouth',(96,128))):
            box=pixel_box(boxes[group],anchors.shape[-1],anchors.shape[-2]);x0,y0,x1,y1=box
            before=anchors[0,:,:,y0:y1,x0:x1].float().contiguous()
            warped=aux['warped'][0,:,:,y0:y1,x0:x1].float().contiguous()
            mask=(self.mask_maker(group,box,anchors.shape[-2:],size,boxes['view_boxes'],self.device).clone()
                  if boxes.get('view_boxes') is not None else soft_mask(*size,self.device).expand(before.shape[0],-1,-1,-1))
            result['patches'][group]=dict(box=box,before=before,
                canonical=F.interpolate(before,size,mode='bilinear',align_corners=False),
                reference=F.interpolate(warped,size,mode='bilinear',align_corners=False),mask=mask)
        return result


    def _brow_forward(self, visible, reference, pose, mask):
        with amp(self.device,self.device.type=='cuda'):
            return self.brow_head(visible,reference,pose,mask)['delta']


    def _rad_forward(self, features, visible, pose, mask):
        with amp(self.device,self.device.type=='cuda'):
            return self.rad_head(features,visible,pose,mask)['delta']


    @torch.no_grad()
    def render(self, raw):
        raw=validate_pose(raw)
        if raw==self._fast_frame_key:
            return self._fast_frame_value
        brows=self.brow_head is not None and any(raw[9:12]) and self._brow_layout is not None
        if self.brow_head is not None and any(raw[9:12]) and self._brow_layout is None:
            raw=tuple(list(raw[:9])+[0.,0.,0.]+list(raw[12:]))
        context=self._updated_context(raw,brows)
        anchors=context['anchors'].clone()
        local_pose=list(raw)
        if brows: local_pose[9:12]=[0.,0.,0.]
        if local_pose[8]>0: local_pose[8]=0.
        pose_tensor=torch.tensor(local_pose,device=self.device,dtype=torch.float32)
        mouth=context['patches']['mouth']
        extracted=None
        if raw[8]>0 or (brows and raw[8]>=0):
            if self.rad_feature_graph is not None:
                extracted=self.rad_feature_graph.forward(mouth['canonical'],mouth['reference'],pose_tensor,mouth['mask'],context['identity'])
            else:
                with amp(self.device,self.device.type=='cuda'):
                    extracted=extract_mouth_features(self.local,mouth['canonical'],mouth['reference'],local_pose,mouth['mask'],context['identity'])
        self._last_skipped_branches=['eyes']
        for group in ('gaze','mouth'):
            if group=='gaze' and branch_is_inactive(group,raw):
                self._last_skipped_branches.append(group);continue
            patch=context['patches']['eyes' if group=='gaze' else 'mouth']
            if group=='mouth' and extracted is not None:
                delta=extracted['legacy_delta']
            elif self.local_graph_runner is not None:
                delta=self.local_graph_runner.forward(group,patch['canonical'],patch['reference'],pose_tensor,
                    mask=patch['mask'],identity=context['identity'])['delta']
            else:
                with amp(self.device,self.device.type=='cuda'):
                    delta=self.local(group,patch['canonical'],patch['reference'],local_pose,
                        mask=patch['mask'],identity=context['identity'])['delta']
            delta=F.interpolate(delta,patch['before'].shape[-2:],mode='bilinear',align_corners=False)
            delta=(patch['before']+delta).clamp(0,1)-patch['before']
            x0,y0,x1,y1=patch['box'];anchors[0,:,:,y0:y1,x0:x1]+=delta
        anchors.clamp_(0,1)
        raw_tensor=torch.tensor(raw,device=self.device,dtype=torch.float32)
        if raw[8]>0:
            x0,y0,x1,y1=mouth['box'];visible=anchors[0,:,:,y0:y1,x0:x1].contiguous()
            small=F.interpolate(visible,mouth['canonical'].shape[-2:],mode='bilinear',align_corners=False)
            delta=self.rad_runner(extracted['features'],small,raw_tensor,mouth['mask'])
            delta=F.interpolate(delta,visible.shape[-2:],mode='bilinear',align_corners=False)
            anchors[0,:,:,y0:y1,x0:x1]+=(visible+delta).clamp(0,1)-visible
            anchors.clamp_(0,1)
        if brows:
            layout=self._tracked_brow_layout
            h,w=anchors.shape[-2:];box=pixel_box(crop_bounds(layout),w,h);x0,y0,x1,y1=box
            before=anchors[0,:,:,y0:y1,x0:x1].contiguous()
            reference=self._brow_warped[0,:,:,y0:y1,x0:x1].float().contiguous()
            size=(self.brow_head.config['patch_height'],self.brow_head.config['patch_width'])
            mask=support_mask(layout,box,(h,w),size,self.device)
            delta=self.brow_runner(F.interpolate(before,size,mode='bilinear',align_corners=False),
                F.interpolate(reference,size,mode='bilinear',align_corners=False),raw_tensor,mask)
            delta=F.interpolate(delta,before.shape[-2:],mode='bilinear',align_corners=False)
            anchors[0,:,:,y0:y1,x0:x1]+=(before+delta).clamp(0,1)-before
        frame=self.base.swizzle(anchors.clamp(0,1))[0]
        self._fast_frame_key,self._fast_frame_value=raw,frame
        return frame

